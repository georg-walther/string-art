using System.Collections.Concurrent;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using ScottPlot;
using System.Numerics;
using System.Drawing.Printing;
using System.Security.Cryptography.X509Certificates;

static Rgba32 Subtract(Rgba32 color1, Rgba32 color2)
{
    byte r = (byte)Math.Max(color1.R - color2.R, 0);
    byte g = (byte)Math.Max(color1.G - color2.G, 0);
    byte b = (byte)Math.Max(color1.B - color2.B, 0);

    return new Rgba32(r, g, b);
}

//Rgba32 testcolor = new Rgba32(20, 20, 20);
//Rgba32 testcolor = new Rgba32(0, 0, 0);

static double Cross(Vector2 vector1, Vector2 vector2)
{
    return vector1.X * vector2.Y - vector1.Y * vector2.X;
}

static float DistanceToLine(Vector2 a, Tuple<Vector2, Vector2> line)
{
    // distance = |(p-q) x (a-q)| / |p-q|
    Vector2 p = line.Item1;
    Vector2 q = line.Item2;

    // Calculate the cross product and its magnitude
    float crossProductMagnitude = (float)Math.Abs(Cross(p - q, a - q));

    // Calculate the distance between p and q
    float distancePQ = Vector2.Distance(p, q);

    // Calculate the distance from a to the line
    float distance = crossProductMagnitude / distancePQ;

    return distance;
}

Vector2 testPoint = new Vector2(7, 7);
Vector2 testLine1 = new Vector2(0, 0);
Vector2 testLine2 = new Vector2(10, 0);
double testDist = DistanceToLine(testPoint, new Tuple<Vector2, Vector2>(testLine1, testLine2));
Console.WriteLine("Distance: " + testDist);

static float DarknessProfile(float x)
{
    float opaqueness = 1;
    float width = 0.25f;
    int grayWidth = 1;
    if (x >= -grayWidth && x < -width)
    {
        // return 0.2 / 1.5 * (x + 2);
        return PointsToLine(x, -grayWidth, 0, -width, opaqueness);
    }
    else if (x >= -width && x < width)
    {
        return opaqueness;
    }
    else if (x >= width && x < grayWidth)
    {
        // return -0.2 / 1.5 * (x - 0.5) + 0.2;
        return PointsToLine(x, width, opaqueness, grayWidth, 0);
    }
    else
    {
        return 0;
    }
}

float[,] kernel =
{
    { 0.5f, 0.5f, 0.5f },
    { 0.5f,  1, 0.5f },
    { 0.5f, 0.5f, 0.5f }
};

static float PointsToLine(float x, float x1, float y1, float x2, float y2)
{
    float m = (y2 - y1) / (x2 - x1);
    float y = m * (x - x1) + y1;
    return y;
}

// Create a new ScottPlot.Plot
var plt = new Plot(600, 400);

// Generate data
double[] xs = DataGen.Range(-3, 3, 0.1);
double[] ys = new double[xs.Length];
for (int i = 0; i < xs.Length; i++)
{
    ys[i] = DarknessProfile((float)xs[i]);
}

// Add the data to the plot
plt.AddScatter(xs, ys);

// Save the plot to a PNG file
plt.SaveFig("darknessProfile.png");

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

static List<Point> BresenhamThickLine(Point p1, Point p2, int thickness)
{
    List<Point> points = new List<Point>();
    int thicknessEachSide = thickness / 2;

    // Calculate direction vector
    Vector2 dirVec = new Vector2(p2.X - p1.X, p2.Y - p1.Y);
    dirVec = Vector2.Normalize(dirVec);
    // Calculate orthogonal vectors
    Vector2 normal = new Vector2(-dirVec.Y, dirVec.X);

    for (int i = -thicknessEachSide; i <= thicknessEachSide; i++)
    {
        // Calculate offset
        Point offset = new Point((int)Math.Round(normal.X * i), (int)Math.Round(normal.Y * i));
        // Calculate points for the parallel line
        Point p1Parallel = new Point(p1.X + offset.X, p1.Y + offset.Y);
        Point p2Parallel = new Point(p2.X + offset.X, p2.Y + offset.Y);

        List<Point> thisPoints = Bresenham(p1Parallel, p2Parallel);
        points.AddRange(thisPoints);
    }
    return points;
}

Point p1 = new Point(0, 0);
Point p2 = new Point(10, 0);
int thickness = 4;
BresenhamThickLine(p1, p2, thickness);

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
using (Image<Rgba32> image = Image.Load<Rgba32>("vikings.jpg"))
{
    // Create a matrix
    int width = image.Width;
    int height = image.Height;
    int[,] matrix = new int[width, height];
    Console.WriteLine("Matrix size: " + width + "x" + height);

    // Fill the matrix
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Rgba32 pixel = image[x, y];
            int luminance = (int)(1 / 3 * pixel.R + 1 / 3 * pixel.G + 1 / 3 * pixel.B);
            matrix[x, y] = luminance;
        }
    }

    Tuple<int, int> center = new Tuple<int, int>(width / 2, height / 2);
    Console.WriteLine("Center: " + center);
    int radius = new int[] { width, height }.Min() / 2;
    Console.WriteLine("Radius: " + radius);

    // white pixels that are further away than the radius
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int dx = center.Item1 - x;
            int dy = center.Item2 - y;
            if (Math.Sqrt(dx * dx + dy * dy) > radius)
            {
                image[x, y] = new Rgba32(255, 255, 255);
                matrix[x, y] = 255;
            }
        }
    }
    image.Save("circle.png");

    // Now you can perform mathematical operations on the matrix
    int numNails = 200;
    double angleStep = 2 * Math.PI / numNails; // because U = 2*pi*r
    List<Tuple<int, int>> nails = new List<Tuple<int, int>>();
    for (int i = 0; i < numNails; i++)
    {
        double angle = i * angleStep;
        int x = center.Item1 + (int)((radius - 1) * Math.Cos(angle));
        int y = center.Item2 + (int)((radius - 1) * Math.Sin(angle));
        nails.Add(new Tuple<int, int>(x, y));
        // Console.WriteLine("Nail " + i + ": " + x + "," + y);
        image[x, y] = new Rgba32(255, 0, 0);
    }
    image.Save("nails.png");


    // Greedy algorithm
    Image<Rgba32> canvas = new Image<Rgba32>(width, height);
    canvas.Mutate(x => x.BackgroundColor(new Rgba32(255, 255, 255)));

    int noChangeCount = 0;
    double prevError = double.MaxValue;
    double threshold = 0.0001;
    int iterations = 0;
    int maxInterations = 1000;
    int currNailIdx = 0;
    int prevNailIdx = -1;
    // while (noChangeCount < 10 && iterations <= maxInterations)
    while (iterations <= maxInterations)
    {
        Console.WriteLine("Iteration " + iterations + ": " + prevError);
        ConcurrentBag<Tuple<int, double, Image<Rgba32>>> results = new ConcurrentBag<Tuple<int, double, Image<Rgba32>>>();
        Parallel.For(0, nails.Count, j =>
        {
            // for (int j = 0; j < nails.Count; j++)
            //{
            if (currNailIdx != j && currNailIdx != prevNailIdx)
            {
                // Draw line on clone canvas
                Image<Rgba32> tmpCanvas = canvas.Clone();
                Point n1 = new Point(nails[currNailIdx].Item1, nails[currNailIdx].Item2);
                Point n2 = new Point(nails[j].Item1, nails[j].Item2);
                List<Point> tmpDrawPoints = BresenhamThickLine(n1, n2, 4);
                for (int k = 0; k < tmpDrawPoints.Count; k++)
                {
                    Vector2 p = new Vector2(tmpDrawPoints[k].X, tmpDrawPoints[k].Y);
                    float dist = DistanceToLine(p, new Tuple<Vector2, Vector2>(new Vector2(n1.X, n1.Y), new Vector2(n2.X, n2.Y)));
                    byte grayscale = (byte)(DarknessProfile(dist) * 3 * 255);
                    // Console.WriteLine("Gray: " + grayscale);
                    if ((int)p.X >= 0 && (int)p.X < tmpCanvas.Width && (int)p.Y >= 0 && (int)p.Y < tmpCanvas.Height)
                    {
                        Rgba32 newColor = new Rgba32(grayscale, grayscale, grayscale);
                        Rgba32 existingColor = tmpCanvas[(int)p.X, (int)p.Y];
                        Rgba32 blendedColor = Subtract(existingColor, newColor);
                        tmpCanvas[(int)p.X, (int)p.Y] = blendedColor;
                    }
                }
                // Calculate error
                double error = CalculateMeanSquaredError(image, tmpCanvas);
                results.Add(new Tuple<int, double, Image<Rgba32>>(j, error, tmpCanvas));
            }
        });
        // Find the nail with the smallest error
        var bestResult = results.OrderBy(t => t.Item2).First();
        int bestNail = bestResult.Item1;
        double minError = bestResult.Item2;
        canvas = bestResult.Item3;
        prevNailIdx = currNailIdx;
        currNailIdx = bestNail;
        if (Math.Abs(minError - prevError) < threshold)
        {
            noChangeCount++;
        }
        prevError = minError;
        iterations++;
        if (iterations == 10)
        {
            canvas.Save("lines-10.png");
        }
    }
    canvas.Save("lines.png");
}