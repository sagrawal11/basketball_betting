import { Trophy, History } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import GameCard from "@/components/GameCard";
import { useEffect, useState } from "react";

// API base URL
const API_BASE = "http://localhost:5001/api";

interface Game {
  id: string;
  homeTeam: { name: string; logo: string; abbrev: string };
  awayTeam: { name: string; logo: string; abbrev: string };
  time: string;
  date: string;
  location: string;
}

const Games = () => {
  const navigate = useNavigate();
  const [games, setGames] = useState<Game[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchGames();
  }, []);

  const fetchGames = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/games/today`);
      const data = await response.json();
      
      if (data.success) {
        setGames(data.games || []);
      } else {
        setError(data.error || 'Failed to load games');
      }
    } catch (err) {
      setError('Failed to connect to API. Make sure Flask server is running on port 5000.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-center relative">
            <h1 className="text-4xl font-black tracking-tight">
              Odds
            </h1>
            
            {/* History Button - Absolute positioned */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => navigate("/history")}
              className="gap-2 text-muted-foreground hover:text-primary hover:bg-primary/5 absolute right-0"
            >
              <History className="w-4 h-4" />
              <span className="hidden sm:inline">History</span>
            </Button>
          </div>
        </div>
      </header>

      {/* Games Grid */}
      <main className="container mx-auto px-4 py-8">
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-muted-foreground mb-1">
            {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
          </h2>
          <p className="text-sm text-muted-foreground">
            {loading ? 'Loading...' : `${games.length} games scheduled`}
          </p>
        </div>

        {error && (
          <div className="bg-destructive/10 border border-destructive/50 text-destructive px-4 py-3 rounded-lg mb-6">
            <p className="font-medium">{error}</p>
            <button onClick={fetchGames} className="text-sm underline mt-2">Try again</button>
          </div>
        )}

        {loading ? (
          <div className="text-center py-12 text-muted-foreground">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
            <p>Loading today's games...</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 animate-fade-in">
            {games.map((game) => (
              <GameCard
                key={game.id}
                gameId={game.id}
                homeTeam={game.homeTeam}
                awayTeam={game.awayTeam}
                time={game.time}
                date={game.date}
                location={game.location}
              />
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default Games;
