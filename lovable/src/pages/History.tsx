import { History as HistoryIcon, ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import HistoricalGameCard from "@/components/HistoricalGameCard";
import { useEffect, useState } from "react";

const API_BASE = "http://localhost:5001/api";

interface HistoricalGame {
  id: string;
  homeTeam: any;
  awayTeam: any;
  time: string;
  date: string;
  location: string;
  accuracy: any;
}

const History = () => {
  const navigate = useNavigate();
  const [games, setGames] = useState<HistoricalGame[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHistoricalGames();
  }, []);

  const fetchHistoricalGames = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/history/games`);
      const data = await response.json();
      
      if (data.success) {
        setGames(data.games || []);
      } else {
        setError(data.error || 'Failed to load historical games');
      }
    } catch (err) {
      setError('Failed to connect to API');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate("/")}
            className="gap-2 text-muted-foreground hover:text-foreground mb-3"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Today's Games
          </Button>
          
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
              <HistoryIcon className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold">Historical Performance</h1>
              <p className="text-sm text-muted-foreground">Past predictions & accuracy metrics</p>
            </div>
          </div>
        </div>
      </header>

      {/* Historical Games */}
      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-muted-foreground mb-1">
            Yesterday's Games
          </h2>
          <p className="text-sm text-muted-foreground">
            {loading ? 'Loading...' : `${games.length} completed games`}
          </p>
        </div>

        {error && (
          <div className="bg-destructive/10 border border-destructive/50 text-destructive px-4 py-3 rounded-lg mb-6">
            <p className="font-medium">{error}</p>
            <button onClick={fetchHistoricalGames} className="text-sm underline mt-2">Try again</button>
          </div>
        )}

        {loading ? (
          <div className="text-center py-12 text-muted-foreground">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
            <p>Loading historical data...</p>
          </div>
        ) : games.length === 0 ? (
          <div className="max-w-2xl mx-auto text-center py-16">
            <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-primary/10 flex items-center justify-center">
              <HistoryIcon className="w-10 h-10 text-primary" />
            </div>
            <h2 className="text-3xl font-bold mb-4">No Games Yet</h2>
            <p className="text-muted-foreground text-lg mb-8">
              Historical data will appear here after games are completed
            </p>
            <Button 
              onClick={() => navigate("/")} 
              className="bg-primary hover:bg-primary/90"
            >
              View Today's Games
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 animate-fade-in">
            {games.map((game) => (
              <HistoricalGameCard
                key={game.id}
                gameId={game.id}
                homeTeam={game.homeTeam}
                awayTeam={game.awayTeam}
                time={game.time}
                date={game.date}
                location={game.location}
                accuracy={game.accuracy}
              />
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default History;
