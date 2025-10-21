import { useNavigate } from "react-router-dom";
import { Card } from "@/components/ui/card";
import { MapPin, Clock } from "lucide-react";

interface GameCardProps {
  gameId: string;
  homeTeam: {
    name: string;
    logo: string;
    abbrev?: string;
  };
  awayTeam: {
    name: string;
    logo: string;
    abbrev?: string;
  };
  time: string;
  date: string;
  location: string;
}

const GameCard = ({ gameId, homeTeam, awayTeam, time, date, location }: GameCardProps) => {
  const navigate = useNavigate();

  const handleClick = () => {
    // Pass team abbreviations in URL for API calls
    const url = `/game/${gameId}?home=${homeTeam.abbrev || 'HOME'}&away=${awayTeam.abbrev || 'AWAY'}`;
    navigate(url);
  };

  return (
    <Card
      onClick={handleClick}
      className="group relative overflow-hidden bg-card border border-border hover:border-primary/50 transition-all duration-300 cursor-pointer hover:scale-[1.02] hover:shadow-[0_0_30px_rgba(0,255,148,0.2)]"
    >
      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
      
      <div className="relative p-6">
        {/* Teams */}
        <div className="flex items-center justify-between gap-8 mb-6">
          {/* Away Team */}
          <div className="flex flex-col items-center gap-3 flex-1">
            <div className="w-20 h-20 rounded-full bg-secondary/50 border-2 border-border group-hover:border-primary/50 transition-all duration-300 flex items-center justify-center overflow-hidden">
              <img 
                src={awayTeam.logo} 
                alt={awayTeam.name}
                className="w-14 h-14 object-contain"
              />
            </div>
            <span className="text-sm font-semibold text-center">{awayTeam.name}</span>
          </div>

          {/* VS Divider */}
          <div className="flex flex-col items-center">
            <span className="text-xs font-bold text-muted-foreground mb-1">VS</span>
            <div className="h-12 w-px bg-gradient-to-b from-transparent via-primary/50 to-transparent" />
          </div>

          {/* Home Team */}
          <div className="flex flex-col items-center gap-3 flex-1">
            <div className="w-20 h-20 rounded-full bg-secondary/50 border-2 border-border group-hover:border-primary/50 transition-all duration-300 flex items-center justify-center overflow-hidden">
              <img 
                src={homeTeam.logo} 
                alt={homeTeam.name}
                className="w-14 h-14 object-contain"
              />
            </div>
            <span className="text-sm font-semibold text-center">{homeTeam.name}</span>
          </div>
        </div>

        {/* Game Details */}
        <div className="space-y-2 pt-4 border-t border-border/50">
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <Clock className="w-3.5 h-3.5" />
            <span>{date} • {time}</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <MapPin className="w-3.5 h-3.5" />
            <span>{location}</span>
          </div>
        </div>

        {/* View Predictions Button Hint */}
        <div className="mt-4 text-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          <span className="text-xs font-medium text-primary">View Predictions →</span>
        </div>
      </div>
    </Card>
  );
};

export default GameCard;
