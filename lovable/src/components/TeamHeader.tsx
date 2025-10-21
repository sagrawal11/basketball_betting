interface TeamHeaderProps {
  homeTeam: {
    name: string;
    logo: string;
    predictedScore: number;
  };
  awayTeam: {
    name: string;
    logo: string;
    predictedScore: number;
  };
  date: string;
  time: string;
  location: string;
}

const TeamHeader = ({ homeTeam, awayTeam, date, time, location }: TeamHeaderProps) => {
  return (
    <div className="bg-card border border-border rounded-lg p-8 mb-8">
      <div className="flex items-center justify-between gap-8 max-w-4xl mx-auto">
        {/* Away Team */}
        <div className="flex flex-col items-center gap-4 flex-1">
          <div className="w-28 h-28 rounded-full bg-secondary/50 border-2 border-primary/30 flex items-center justify-center overflow-hidden shadow-[0_0_20px_rgba(0,255,148,0.2)]">
            <img 
              src={awayTeam.logo} 
              alt={awayTeam.name}
              className="w-20 h-20 object-contain"
            />
          </div>
          <div className="text-center">
            <h2 className="text-2xl font-bold mb-1">{awayTeam.name}</h2>
            <div className="text-4xl font-black text-primary">{awayTeam.predictedScore}</div>
            <p className="text-xs text-muted-foreground mt-1">Predicted Score</p>
          </div>
        </div>

        {/* Center Info */}
        <div className="flex flex-col items-center gap-3">
          <div className="text-center space-y-1">
            <div className="text-xs font-bold text-muted-foreground uppercase tracking-wider">Game Info</div>
            <div className="text-sm text-foreground/80">{date}</div>
            <div className="text-sm text-foreground/80">{time}</div>
            <div className="text-xs text-muted-foreground">{location}</div>
          </div>
          <div className="h-16 w-px bg-gradient-to-b from-transparent via-primary/50 to-transparent" />
          <div className="text-2xl font-bold text-primary">VS</div>
        </div>

        {/* Home Team */}
        <div className="flex flex-col items-center gap-4 flex-1">
          <div className="w-28 h-28 rounded-full bg-secondary/50 border-2 border-primary/30 flex items-center justify-center overflow-hidden shadow-[0_0_20px_rgba(0,255,148,0.2)]">
            <img 
              src={homeTeam.logo} 
              alt={homeTeam.name}
              className="w-20 h-20 object-contain"
            />
          </div>
          <div className="text-center">
            <h2 className="text-2xl font-bold mb-1">{homeTeam.name}</h2>
            <div className="text-4xl font-black text-primary">{homeTeam.predictedScore}</div>
            <p className="text-xs text-muted-foreground mt-1">Predicted Score</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TeamHeader;
