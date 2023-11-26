public class CoordVertex
{
    public int Id { get; set; }
    public int X { get; set; }
    public int Y { get; set; }

    public CoordVertex(int id, int x, int y)
    {
        Id = id;
        X = x;
        Y = y;
    }

    public bool CoordEquals(object obj)
    {
        if (obj == null || GetType() != obj.GetType())
        {
            return false;
        }

        var other = (CoordVertex)obj;
        return X == other.X && Y == other.Y;
    }
}