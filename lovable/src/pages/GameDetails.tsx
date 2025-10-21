import { useParams, useNavigate, useLocation } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import TeamHeader from "@/components/TeamHeader";
import PlayerCard from "@/components/PlayerCard";
import { useEffect, useState } from "react";

// API base URL
const API_BASE = "http://localhost:5001/api";

interface GameData {
  homeTeam: { name: string; logo: string; predictedScore: number };
  awayTeam: { name: string; logo: string; predictedScore: number };
  date: string;
  time: string;
  location: string;
  homePlayers: any[];
  awayPlayers: any[];
}

const GameDetails = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [gameData, setGameData] = useState<GameData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Get team abbreviations from URL params
  const searchParams = new URLSearchParams(location.search);
  const homeAbbrev = searchParams.get('home');
  const awayAbbrev = searchParams.get('away');

  useEffect(() => {
    if (id && homeAbbrev && awayAbbrev) {
      fetchGameData();
    }
  }, [id, homeAbbrev, awayAbbrev]);

  const fetchGameData = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE}/game/${id}/players?home=${homeAbbrev}&away=${awayAbbrev}`);
      const data = await response.json();
      
      if (data.success) {
        setGameData(data);
      } else {
        setError(data.error || 'Failed to load game data');
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
            className="gap-2 text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Games
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 animate-fade-in">
        {loading ? (
          <div className="text-center py-12 text-muted-foreground">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary mb-4"></div>
            <p>Loading game data and generating predictions...</p>
            <p className="text-sm mt-2">This may take 10-15 seconds</p>
          </div>
        ) : error ? (
          <div className="bg-destructive/10 border border-destructive/50 text-destructive px-4 py-3 rounded-lg">
            <p className="font-medium">{error}</p>
            <button onClick={fetchGameData} className="text-sm underline mt-2">Try again</button>
          </div>
        ) : gameData ? (
          <>
            {/* Team Header */}
            <TeamHeader
              homeTeam={gameData.homeTeam}
              awayTeam={gameData.awayTeam}
              date={gameData.date}
              time={gameData.time}
              location={gameData.location}
            />

            {/* Players Grid */}
            <div className="grid lg:grid-cols-2 gap-8">
              {/* Away Team Players */}
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-full bg-secondary/50 border border-border flex items-center justify-center overflow-hidden">
                    <img 
                      src={gameData.awayTeam.logo} 
                      alt={gameData.awayTeam.name}
                      className="w-7 h-7 object-contain"
                    />
                  </div>
                  <h3 className="text-lg font-bold">{gameData.awayTeam.name} Players</h3>
                </div>
                <div className="grid gap-4">
                  {gameData.awayPlayers.map((player) => (
                    <PlayerCard key={player.name} {...player} />
                  ))}
                </div>
              </div>

              {/* Home Team Players */}
              <div>
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-10 h-10 rounded-full bg-secondary/50 border border-border flex items-center justify-center overflow-hidden">
                    <img 
                      src={gameData.homeTeam.logo} 
                      alt={gameData.homeTeam.name}
                      className="w-7 h-7 object-contain"
                    />
                  </div>
                  <h3 className="text-lg font-bold">{gameData.homeTeam.name} Players</h3>
                </div>
                <div className="grid gap-4">
                  {gameData.homePlayers.map((player) => (
                    <PlayerCard key={player.name} {...player} />
                  ))}
                </div>
              </div>
            </div>
          </>
        ) : null}
      </main>
    </div>
  );
};

export default GameDetails;
