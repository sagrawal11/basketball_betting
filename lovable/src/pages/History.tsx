import { History as HistoryIcon, ArrowLeft } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";

const History = () => {
  const navigate = useNavigate();

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

      {/* Empty State */}
      <main className="container mx-auto px-4 py-16">
        <div className="max-w-2xl mx-auto text-center">
          <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-primary/10 flex items-center justify-center">
            <HistoryIcon className="w-10 h-10 text-primary" />
          </div>
          <h2 className="text-3xl font-bold mb-4">No Historical Data Yet</h2>
          <p className="text-muted-foreground text-lg mb-8">
            Historical performance tracking will begin once games are completed. 
            After each game, we'll compare our predictions to actual results and display accuracy metrics here.
          </p>
          <div className="bg-card border border-border rounded-lg p-6 text-left">
            <h3 className="font-semibold mb-3 text-primary">Coming Soon:</h3>
            <ul className="space-y-2 text-sm text-muted-foreground">
              <li>✓ Predicted vs Actual scores for each game</li>
              <li>✓ Individual player stat prediction accuracy</li>
              <li>✓ Overall model performance metrics (MAE, R², MAPE)</li>
              <li>✓ Betting recommendation success rate</li>
              <li>✓ Historical trends and improvements over time</li>
            </ul>
          </div>
          <Button 
            onClick={() => navigate("/")} 
            className="mt-8 bg-primary hover:bg-primary/90"
          >
            View Today's Games
          </Button>
        </div>
      </main>
    </div>
  );
};

export default History;
