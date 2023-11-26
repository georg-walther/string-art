using System.Collections.Concurrent;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using ScottPlot;
using System.Numerics;
using QuickGraph;


float[,] kernel =
{
    { 0.5f, 0.5f, 0.5f },
    { 0.5f,  1, 0.5f },
    { 0.5f, 0.5f, 0.5f }
};

static List<Point> Bresenham(Point p1, Point p2)
{
    List<Point> points = new List<Point>();
    int dx = Math.Abs(p2.X - p1.X), sx = p1.X < p2.X ? 1 : -1;
    int dy = -Math.Abs(p2.Y - p1.Y), sy = p1.Y < p2.Y ? 1 : -1;
    int err = dx + dy, e2;

    while (true)
    {
        points.Add(new Point(p1.X, p1.Y));
        if (p1.X == p2.X && p1.Y == p2.Y) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; p1.X += sx; }
        if (e2 <= dx) { err += dx; p1.Y += sy; }
    }
    return points;
}

static Image<Rgba32> drawEdgesOnCanvas(AdjacencyGraph<CoordVertex, Edge<CoordVertex>> graph, Image<Rgba32> canvas)
{
    foreach (Edge<CoordVertex> edge in graph.Edges)
    {
        Point n1 = new(edge.Source.X, edge.Source.Y);
        Point n2 = new(edge.Target.X, edge.Target.Y);
        List<Point> drawPoints = Bresenham(n1, n2);
        for (int i = 0; i < drawPoints.Count; i++)
        {
            Vector2 p = new(drawPoints[i].X, drawPoints[i].Y);
            canvas[(int)p.X, (int)p.Y] = new Rgba32(0, 0, 0); // paint black
        }
    }
    return canvas;
}

static AdjacencyGraph<CoordVertex, Edge<CoordVertex>> downSampleGraphCoords(AdjacencyGraph<CoordVertex, Edge<CoordVertex>> graph, float factor)
{
    foreach (CoordVertex vertex in graph.Vertices)
    {
        vertex.X = (int)((float)vertex.X * factor);
        vertex.Y = (int)((float)vertex.Y * factor);
    }
    return graph;
}

static double CalculateMeanSquaredError(Image<Rgba32> img1, Image<Rgba32> img2)
{
    if (img1.Width != img2.Width || img1.Height != img2.Height)
    {
        throw new ArgumentException("Images are not the same size.");
    }

    double mse = 0.0;

    for (int y = 0; y < img1.Height; y++)
    {
        for (int x = 0; x < img1.Width; x++)
        {
            Rgba32 pixel1 = img1[x, y];
            Rgba32 pixel2 = img2[x, y];

            mse += Math.Pow(pixel1.R - pixel2.R, 2);
            mse += Math.Pow(pixel1.G - pixel2.G, 2);
            mse += Math.Pow(pixel1.B - pixel2.B, 2);
        }
    }

    return mse / (img1.Width * img1.Height * 3);
}

Console.WriteLine(Directory.GetCurrentDirectory());

// Load the image
Image<Rgba32> img = Image.Load<Rgba32>("Marie_Curie.jpg");
int width = img.Width;
int height = img.Height;

Tuple<int, int> center = new Tuple<int, int>(width / 2, height / 2);
Console.WriteLine("Center: " + center);
int radius = new int[] { width, height }.Min() / 2;
Console.WriteLine("Radius: " + radius);

int size = 2 * radius;
Rectangle rect = new(center.Item1 - radius, center.Item2 - radius, size, size);
img.Mutate(i => i.Crop(rect));
center = new Tuple<int, int>(size / 2, size / 2);

for (int x = 0; x < size; x++)
{
    for (int y = 0; y < size; y++)
    {
        int dx = center.Item1 - x;
        int dy = center.Item2 - y;
        if (Math.Sqrt(dx * dx + dy * dy) > radius)
        {
            // Paint pixels white that are further away than the radius
            img[x, y] = new Rgba32(255, 255, 255);
        }
    }
}

// Calculate nail positions
var graph = new AdjacencyGraph<CoordVertex, Edge<CoordVertex>>();
int numNails = 200;
double angleStep = 2 * Math.PI / numNails; // because U = 2*pi*r
for (int i = 0; i < numNails; i++)
{
    double angle = i * angleStep;
    int x = center.Item1 + (int)((radius - 1) * Math.Cos(angle));
    int y = center.Item2 + (int)((radius - 1) * Math.Sin(angle));
    graph.AddVertex(new CoordVertex(i, x, y));
    // Console.WriteLine("Nail " + i + ": " + x + "," + y);
    img[x, y] = new Rgba32(255, 0, 0);
}
img.Save("nails.png");

// Create canvas
Image<Rgba32> canvas = new Image<Rgba32>(size, size);
canvas.Mutate(x => x.BackgroundColor(new Rgba32(255, 255, 255)));
// Greedy algorithm
int currNailId = 0;
int prevNailId = -1;

int lowestRes = 32; // 32 x 32 pixel
int iteration = 0;
int maxInter = 1000;
int fullResIter = 300;
while (iteration <= maxInter)
{
    // Increase resolution per iteration.
    // Low resolution to speed up the algorithm at the beginning where the error minimization is "easy".
    float fullResFraction = Math.Min(1, (float)iteration / (maxInter - fullResIter));
    int resolution = (int)((fullResFraction * (size - lowestRes)) + lowestRes);
    Console.WriteLine("Resolution: " + resolution);
    Image<Rgba32> lowResImg = img.Clone();
    // lowResImg.Mutate(x => x.Resize(resolution, resolution));
    Image<Rgba32> lowResCanvas = canvas.Clone();
    // lowResCanvas.Mutate(x => x.Resize(resolution, resolution));

    ConcurrentBag<Tuple<int, double, AdjacencyGraph<CoordVertex, Edge<CoordVertex>>>> results = new();
    Parallel.For(0, graph.VertexCount, j =>
    {
        // for (int j = 0; j < graph.VertexCount; j++)
        // {
        if (currNailId != j && currNailId != prevNailId) // don't draw line to previous nail or same nail
        {
            // Draw line on clone canvas
            Image<Rgba32> tmpCanvas = lowResCanvas.Clone();
            var tmpGraph = graph.Clone();
            tmpGraph.AddEdge(new Edge<CoordVertex>(tmpGraph.Vertices.ElementAt(currNailId), tmpGraph.Vertices.ElementAt(j)));
            // AdjacencyGraph<CoordVertex, Edge<CoordVertex>> debugGraph = downSampleGraphCoords(tmpGraph.Clone(), (float)resolution / size);

            tmpCanvas = drawEdgesOnCanvas(tmpGraph, tmpCanvas);
            // tmpCanvas = drawEdgesOnCanvas(debugGraph, tmpCanvas);
            // Calculate error
            tmpCanvas.Mutate(x => x.GaussianBlur(0.5f));
            double error = CalculateMeanSquaredError(lowResImg, tmpCanvas);
            results.Add(new Tuple<int, double, AdjacencyGraph<CoordVertex, Edge<CoordVertex>>>(j, error, tmpGraph));
        }
    });
    //}
    // Find the nail with the smallest error
    var bestResult = results.OrderBy(t => t.Item2).First();
    int bestNail = bestResult.Item1;
    double minError = bestResult.Item2;
    graph = bestResult.Item3;
    prevNailId = currNailId; // update previous nail 
    currNailId = bestNail; // update current nail

    Console.WriteLine("Iteration " + iteration + ": " + minError);
    iteration++;
    if (iteration % 10 == 0)
    {
        Image<Rgba32> notFinishedCanvas = drawEdgesOnCanvas(graph, canvas.Clone());
        notFinishedCanvas.Mutate(x => x.GaussianBlur(0.5f));
        notFinishedCanvas.Save("curr_lines.png");
    }
}
canvas = drawEdgesOnCanvas(graph, canvas);
canvas.Mutate(x => x.GaussianBlur(0.5f));
canvas.Save("lines.png");