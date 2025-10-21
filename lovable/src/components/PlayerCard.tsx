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
}

const PlayerCard = ({ name, position, image, stats }: PlayerCardProps) => {
  return (
    <Card className="group bg-card border border-border hover:border-primary/50 transition-all duration-300 overflow-hidden hover:scale-[1.02]">
      <div className="p-4">
        {/* Player Header */}
        <div className="flex items-center gap-4 mb-4">
          <div className="relative">
            <div className="w-16 h-16 rounded-full bg-secondary/50 border-2 border-border group-hover:border-primary/50 transition-all duration-300 overflow-hidden flex items-center justify-center">
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
            <h4 className="font-semibold text-sm truncate">{name}</h4>
            <p className="text-xs text-muted-foreground">{position}</p>
          </div>
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
