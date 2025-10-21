import { useNavigate } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { MapPin, Clock, TrendingUp, TrendingDown } from "lucide-react";

interface HistoricalGameCardProps {
  gameId: string;
  homeTeam: {
    name: string;
    logo: string;
    predictedScore: number;
    actualScore: number;
  };
  awayTeam: {
    name: string;
    logo: string;
    predictedScore: number;
    actualScore: number;
  };
  time: string;
  date: string;
  location: string;
  accuracy: {
    scoreAccuracy: number; // percentage
    playerStatsAccuracy: number; // percentage
    avgPointsDiff: number;
  };
}

const HistoricalGameCard = ({ 
  gameId, 
  homeTeam, 
  awayTeam, 
  time, 
  date, 
  location, 
  accuracy 
}: HistoricalGameCardProps) => {
  const navigate = useNavigate();
  
  const totalScoreDiff = Math.abs(
    (homeTeam.predictedScore - homeTeam.actualScore) + 
    (awayTeam.predictedScore - awayTeam.actualScore)
  );

  const getAccuracyColor = (acc: number) => {
    if (acc >= 90) return "text-primary";
    if (acc >= 75) return "text-primary/70";
    return "text-muted-foreground";
  };

  return (
    <Card
      onClick={() => navigate(`/game/${gameId}`)}
      className="group relative overflow-hidden bg-card border border-border hover:border-primary/50 transition-all duration-300 cursor-pointer hover:scale-[1.02] hover:shadow-[0_0_30px_rgba(0,255,148,0.2)]"
    >
      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      
      <div className="relative p-6">
        {/* Teams */}
        <div className="flex items-center justify-between gap-6 mb-4">
          {/* Away Team */}
          <div className="flex flex-col items-center gap-2 flex-1">
            <div className="w-16 h-16 rounded-full bg-secondary/50 border-2 border-border group-hover:border-primary/50 transition-all duration-300 flex items-center justify-center overflow-hidden">
              <img 
                src={awayTeam.logo} 
                alt={awayTeam.name}
                className="w-11 h-11 object-contain"
              />
            </div>
            <span className="text-xs font-semibold text-center">{awayTeam.name}</span>
            
            {/* Scores */}
            <div className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground">Pred: {awayTeam.predictedScore}</span>
              <span className="text-foreground font-bold">Act: {awayTeam.actualScore}</span>
            </div>
            <div className="text-[10px] text-muted-foreground">
              {awayTeam.predictedScore > awayTeam.actualScore ? (
                <span className="flex items-center gap-1">
                  <TrendingDown className="w-3 h-3" />
                  Off by {Math.abs(awayTeam.predictedScore - awayTeam.actualScore)}
                </span>
              ) : awayTeam.predictedScore < awayTeam.actualScore ? (
                <span className="flex items-center gap-1">
                  <TrendingUp className="w-3 h-3" />
                  Off by {Math.abs(awayTeam.predictedScore - awayTeam.actualScore)}
                </span>
              ) : (
                <span className="text-primary">Perfect!</span>
              )}
            </div>
          </div>

          {/* VS Divider */}
          <div className="flex flex-col items-center">
            <span className="text-[10px] font-bold text-muted-foreground mb-1">VS</span>
            <div className="h-8 w-px bg-gradient-to-b from-transparent via-primary/50 to-transparent" />
          </div>

          {/* Home Team */}
          <div className="flex flex-col items-center gap-2 flex-1">
            <div className="w-16 h-16 rounded-full bg-secondary/50 border-2 border-border group-hover:border-primary/50 transition-all duration-300 flex items-center justify-center overflow-hidden">
              <img 
                src={homeTeam.logo} 
                alt={homeTeam.name}
                className="w-11 h-11 object-contain"
              />
            </div>
            <span className="text-xs font-semibold text-center">{homeTeam.name}</span>
            
            {/* Scores */}
            <div className="flex items-center gap-2 text-xs">
              <span className="text-muted-foreground">Pred: {homeTeam.predictedScore}</span>
              <span className="text-foreground font-bold">Act: {homeTeam.actualScore}</span>
            </div>
            <div className="text-[10px] text-muted-foreground">
              {homeTeam.predictedScore > homeTeam.actualScore ? (
                <span className="flex items-center gap-1">
                  <TrendingDown className="w-3 h-3" />
                  Off by {Math.abs(homeTeam.predictedScore - homeTeam.actualScore)}
                </span>
              ) : homeTeam.predictedScore < homeTeam.actualScore ? (
                <span className="flex items-center gap-1">
                  <TrendingUp className="w-3 h-3" />
                  Off by {Math.abs(homeTeam.predictedScore - homeTeam.actualScore)}
                </span>
              ) : (
                <span className="text-primary">Perfect!</span>
              )}
            </div>
          </div>
        </div>

        {/* Accuracy Metrics */}
        <div className="grid grid-cols-3 gap-2 mb-3 pt-3 border-t border-border/50">
          <div className="text-center">
            <div className={`text-sm font-bold ${getAccuracyColor(accuracy.scoreAccuracy)}`}>
              {accuracy.scoreAccuracy}%
            </div>
            <div className="text-[10px] text-muted-foreground">Score Accuracy</div>
          </div>
          <div className="text-center">
            <div className={`text-sm font-bold ${getAccuracyColor(accuracy.playerStatsAccuracy)}`}>
              {accuracy.playerStatsAccuracy}%
            </div>
            <div className="text-[10px] text-muted-foreground">Player Stats</div>
          </div>
          <div className="text-center">
            <div className="text-sm font-bold text-muted-foreground">
              ±{accuracy.avgPointsDiff.toFixed(1)}
            </div>
            <div className="text-[10px] text-muted-foreground">Avg Diff</div>
          </div>
        </div>

        {/* Game Details */}
        <div className="space-y-1 pt-2 border-t border-border/50">
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
            <Clock className="w-3 h-3" />
            <span>{date} • {time}</span>
          </div>
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
            <MapPin className="w-3 h-3" />
            <span>{location}</span>
          </div>
        </div>

        {/* View Details Hint */}
        <div className="mt-3 text-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          <span className="text-[10px] font-medium text-primary">View Full Analysis →</span>
        </div>
      </div>
    </Card>
  );
};

export default HistoricalGameCard;
