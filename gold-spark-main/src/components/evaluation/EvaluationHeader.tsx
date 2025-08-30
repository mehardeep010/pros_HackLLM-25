import { Card } from "@/components/ui/card";

const EvaluationHeader = () => {
  return (
    <Card className="card-gradient border-0 shadow-lg p-8 mb-8">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-gradient">LLM Evaluation Interface</h1>
        <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
          Compare LLM outputs with ground truth references and analyze fidelity, hallucination risk, 
          and concept-level alignment with advanced evaluation metrics.
        </p>
        <div className="flex items-center justify-center gap-4 pt-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-success"></div>
            <span className="text-sm text-muted-foreground">Matching Concepts</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-warning"></div>
            <span className="text-sm text-muted-foreground">Neutral Analysis</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-destructive"></div>
            <span className="text-sm text-muted-foreground">Conflicting Concepts</span>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default EvaluationHeader;