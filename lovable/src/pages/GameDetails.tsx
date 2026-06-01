import { useParams, useNavigate, useLocation } from "react-router-dom";
import { ArrowLeft, AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import TeamHeader from "@/components/TeamHeader";
import PlayerCard from "@/components/PlayerCard";
import { useEffect, useState, useCallback } from "react";
import { API_BASE } from "@/lib/api";

interface InjuryEntry {
  name: string;
  team: string;
  is_home: boolean;
  status: string;
  injury_type: string;
  detail: string;
  side: string;
  headshot_url: string;
}

interface PlayerData {
  name: string;
  position: string;
  image: string;
  stats: {
    points: number;
    rebounds: number;
    assists: number;
    steals: number;
    blocks: number;
    fg: string;
    threePt: string;
    ft: string;
  };
  injuryStatus?: string;
  injuryDetail?: string;
}

interface GameData {
  homeTeam: { name: string; logo: string; predictedScore: number };
  awayTeam: { name: string; logo: string; predictedScore: number };
  date: string;
  time: string;
  location: string;
  homePlayers: PlayerData[];
  awayPlayers: PlayerData[];
  injuryReport: InjuryEntry[];
}

const GameDetails = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const [gameData, setGameData] = useState<GameData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [excludedPlayers, setExcludedPlayers] = useState<Set<string>>(new Set());
  const [refetching, setRefetching] = useState(false);

  // Get team abbreviations from URL params
  const searchParams = new URLSearchParams(location.search);
  const homeAbbrev = searchParams.get('home');
  const awayAbbrev = searchParams.get('away');

  const fetchGameData = useCallback(async (excluded: Set<string> = new Set()) => {
    if (!id || !homeAbbrev || !awayAbbrev) return;

    try {
      if (excluded.size === 0) {
        setLoading(true);
      } else {
        setRefetching(true);
      }

      let url = `${API_BASE}/game/${id}/players?home=${homeAbbrev}&away=${awayAbbrev}`;
      if (excluded.size > 0) {
        const excludeParam = Array.from(excluded).join(",");
        url += `&excludePlayers=${encodeURIComponent(excludeParam)}`;
      }

      const response = await fetch(url);
      const data = await response.json();
      
      if (data.success) {
        setGameData(data);
        setError(null);
      } else {
        setError(data.error || 'Failed to load game data');
      }
    } catch (err) {
      setError('Failed to connect to API');
      console.error(err);
    } finally {
      setLoading(false);
      setRefetching(false);
    }
  }, [id, homeAbbrev, awayAbbrev]);

  useEffect(() => {
    fetchGameData(excludedPlayers);
  }, []);

  const handleToggleExclude = (playerName: string) => {
    setExcludedPlayers(prev => {
      const next = new Set(prev);
      if (next.has(playerName)) {
        next.delete(playerName);
      } else {
        next.add(playerName);
      }
      // Re-fetch with updated exclusions
      fetchGameData(next);
      return next;
    });
  };

  const injuryReport = gameData?.injuryReport || [];
  const homeInjuries = injuryReport.filter(i => i.is_home);
  const awayInjuries = injuryReport.filter(i => !i.is_home);

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate("/")}
            className="gap-2 text-muted-foreground hover:text-foreground"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Games
          </Button>
          {refetching && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground animate-pulse">
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              Recalculating...
            </div>
          )}
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
            <button onClick={() => fetchGameData(excludedPlayers)} className="text-sm underline mt-2">Try again</button>
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

            {/* Injury Report */}
            {injuryReport.length > 0 && (
              <div className="mb-8 bg-card border border-border/50 rounded-lg overflow-hidden">
                <div className="px-4 py-3 bg-red-500/10 border-b border-red-500/20 flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-red-400" />
                  <h3 className="text-sm font-bold text-red-400">Injury Report</h3>
                  <span className="text-xs text-red-400/60 ml-auto">
                    {injuryReport.length} player{injuryReport.length !== 1 ? 's' : ''} sidelined
                  </span>
                </div>
                <div className="p-4 grid sm:grid-cols-2 gap-3">
                  {/* Away injuries */}
                  {awayInjuries.length > 0 && (
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <img 
                          src={gameData.awayTeam.logo} 
                          alt={gameData.awayTeam.name}
                          className="w-5 h-5 object-contain"
                        />
                        <span className="text-xs font-semibold text-muted-foreground">{gameData.awayTeam.name}</span>
                      </div>
                      {awayInjuries.map((inj) => (
                        <InjuryRow key={inj.name} injury={inj} />
                      ))}
                    </div>
                  )}
                  {/* Home injuries */}
                  {homeInjuries.length > 0 && (
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <img 
                          src={gameData.homeTeam.logo} 
                          alt={gameData.homeTeam.name}
                          className="w-5 h-5 object-contain"
                        />
                        <span className="text-xs font-semibold text-muted-foreground">{gameData.homeTeam.name}</span>
                      </div>
                      {homeInjuries.map((inj) => (
                        <InjuryRow key={inj.name} injury={inj} />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Toggle Hint */}
            <div className="mb-6 flex items-center gap-2 text-xs text-muted-foreground bg-secondary/20 rounded-lg px-3 py-2 border border-border/30">
              <span className="text-primary font-medium">💡 What-If Mode:</span>
              <span>Toggle players out using the switches on their cards to see how predictions change</span>
            </div>

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
                  <span className="text-xs text-muted-foreground ml-auto">
                    {gameData.awayPlayers.length} active
                  </span>
                </div>
                <div className="grid gap-4">
                  {gameData.awayPlayers.map((player) => (
                    <PlayerCard
                      key={player.name}
                      {...player}
                      excluded={excludedPlayers.has(player.name)}
                      onToggleExclude={() => handleToggleExclude(player.name)}
                    />
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
                  <span className="text-xs text-muted-foreground ml-auto">
                    {gameData.homePlayers.length} active
                  </span>
                </div>
                <div className="grid gap-4">
                  {gameData.homePlayers.map((player) => (
                    <PlayerCard
                      key={player.name}
                      {...player}
                      excluded={excludedPlayers.has(player.name)}
                      onToggleExclude={() => handleToggleExclude(player.name)}
                    />
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

const InjuryRow = ({ injury }: { injury: InjuryEntry }) => {
  const injuryText = [injury.injury_type, injury.detail, injury.side]
    .filter(Boolean)
    .filter(s => s !== "Not Specified")
    .join(" · ");

  return (
    <div className="flex items-center gap-3 py-2 px-2 rounded-md hover:bg-secondary/20 transition-colors">
      {injury.headshot_url ? (
        <img
          src={injury.headshot_url}
          alt={injury.name}
          className="w-8 h-8 rounded-full object-cover bg-secondary/50"
        />
      ) : (
        <div className="w-8 h-8 rounded-full bg-secondary/50" />
      )}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold truncate">{injury.name}</span>
          <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full border bg-red-500/20 text-red-400 border-red-500/30">
            OUT
          </span>
        </div>
        {injuryText && (
          <p className="text-[10px] text-muted-foreground truncate">{injuryText}</p>
        )}
      </div>
    </div>
  );
};

export default GameDetails;
