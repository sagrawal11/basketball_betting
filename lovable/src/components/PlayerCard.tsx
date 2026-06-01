import { Card } from "@/components/ui/card";

interface PlayerCardProps {
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
  excluded?: boolean;
  onToggleExclude?: () => void;
}

const InjuryBadge = ({ status }: { status: string }) => {
  const lower = status.toLowerCase();
  let color = "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
  let label = status;

  if (lower === "out") {
    color = "bg-red-500/20 text-red-400 border-red-500/30";
    label = "OUT";
  } else if (lower.includes("day")) {
    color = "bg-yellow-500/20 text-yellow-400 border-yellow-500/30";
    label = "GTD";
  } else if (lower.includes("probable")) {
    color = "bg-green-500/20 text-green-400 border-green-500/30";
    label = "PROB";
  }

  return (
    <span className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full border ${color}`}>
      {label}
    </span>
  );
};

const PlayerCard = ({ name, position, image, stats, injuryStatus, injuryDetail, excluded, onToggleExclude }: PlayerCardProps) => {
  return (
    <Card className={`group bg-card border border-border transition-all duration-300 overflow-hidden ${
      excluded
        ? "opacity-40 border-red-500/30 scale-[0.98]"
        : "hover:border-primary/50 hover:scale-[1.02]"
    }`}>
      <div className="p-4">
        {/* Player Header */}
        <div className="flex items-center gap-4 mb-4">
          <div className="relative">
            <div className={`w-16 h-16 rounded-full bg-secondary/50 border-2 transition-all duration-300 overflow-hidden flex items-center justify-center ${
              excluded ? "border-red-500/30" : "border-border group-hover:border-primary/50"
            }`}>
              <img 
                src={image} 
                alt={name}
                className="w-full h-full object-cover"
              />
            </div>
            <div className="absolute -bottom-1 -right-1 bg-primary text-primary-foreground text-[10px] font-bold px-1.5 py-0.5 rounded-full">
              {position}
            </div>
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h4 className="font-semibold text-sm truncate">{name}</h4>
              {injuryStatus && <InjuryBadge status={injuryStatus} />}
            </div>
            <p className="text-xs text-muted-foreground">
              {injuryDetail || position}
            </p>
          </div>
          {onToggleExclude && (
            <button
              onClick={(e) => { e.stopPropagation(); onToggleExclude(); }}
              className={`shrink-0 w-10 h-5 rounded-full transition-all duration-200 relative ${
                excluded
                  ? "bg-red-500/40"
                  : "bg-secondary/60 hover:bg-secondary"
              }`}
              title={excluded ? "Include player" : "Exclude player (what-if)"}
            >
              <div className={`absolute top-0.5 w-4 h-4 rounded-full transition-all duration-200 ${
                excluded
                  ? "right-0.5 bg-red-400"
                  : "left-0.5 bg-muted-foreground/50"
              }`} />
            </button>
          )}
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-3 gap-2">
          <StatItem label="PTS" value={stats.points} />
          <StatItem label="REB" value={stats.rebounds} />
          <StatItem label="AST" value={stats.assists} />
          <StatItem label="STL" value={stats.steals} />
          <StatItem label="BLK" value={stats.blocks} />
          <StatItem label="FG%" value={stats.fg} />
        </div>

        {/* Shooting Stats */}
        <div className="grid grid-cols-2 gap-2 mt-2 pt-2 border-t border-border/50">
          <StatItem label="3PT%" value={stats.threePt} small />
          <StatItem label="FT%" value={stats.ft} small />
        </div>
      </div>
    </Card>
  );
};

const StatItem = ({ label, value, small = false }: { label: string; value: number | string; small?: boolean }) => {
  return (
    <div className={`bg-secondary/30 rounded-lg ${small ? 'p-1.5' : 'p-2'} text-center`}>
      <div className={`font-bold text-primary ${small ? 'text-xs' : 'text-sm'}`}>
        {value}
      </div>
      <div className={`text-muted-foreground ${small ? 'text-[10px]' : 'text-xs'} mt-0.5`}>
        {label}
      </div>
    </div>
  );
};

export default PlayerCard;
