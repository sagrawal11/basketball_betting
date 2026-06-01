import { Trophy, History } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import GameCard from "@/components/GameCard";
import { useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

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
      setError('Failed to connect to API. Make sure Flask server is running on port 5001.');
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
            <h1 className="text-4xl font-black tracking-tight flex items-center gap-3">
              <Trophy className="w-8 h-8 text-primary" />
              Odds
            </h1>
            
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

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 animate-fade-in">
        <div className="mb-6 flex justify-between items-end">
          <div>
            <h2 className="text-xl font-bold tracking-tight mb-1">
              Tonight's Slate
            </h2>
            <p className="text-sm text-muted-foreground">
              {loading ? 'Loading schedule...' : `${games.length} games scheduled`}
            </p>
          </div>
        </div>

        {error && (
          <div className="bg-destructive/10 border border-destructive/50 text-destructive px-4 py-3 rounded-lg mb-6 flex justify-between items-center">
            <p className="font-medium">{error}</p>
            <Button variant="outline" size="sm" onClick={fetchGames}>Try again</Button>
          </div>
        )}

        {loading ? (
          <div className="text-center py-20 text-muted-foreground">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
            <p>Fetching Live Schedule from ESPN...</p>
          </div>
        ) : games.length === 0 && !error ? (
          <div className="text-center py-20 bg-card border border-border/50 rounded-2xl">
            <Trophy className="w-12 h-12 text-muted-foreground mx-auto mb-4 opacity-50" />
            <h3 className="text-lg font-bold mb-1">No Games Tonight</h3>
            <p className="text-muted-foreground">Check back tomorrow for the next slate of NBA games.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
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
