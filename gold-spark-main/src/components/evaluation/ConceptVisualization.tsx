import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { BarChart3, PieChart, TrendingUp } from "lucide-react";

interface ConceptVisualizationProps {
  matching: string[];
  conflicting: string[];
  hypothesisConcepts: string[];
  referenceConcepts: string[];
  fidelityScore: number;
}

const ConceptVisualization = ({ 
  matching, 
  conflicting, 
  hypothesisConcepts, 
  referenceConcepts, 
  fidelityScore 
}: ConceptVisualizationProps) => {
  const totalConcepts = hypothesisConcepts.length + referenceConcepts.length;
  const matchPercentage = totalConcepts > 0 ? (matching.length / totalConcepts) * 100 : 0;
  const conflictPercentage = totalConcepts > 0 ? (conflicting.length / totalConcepts) * 100 : 0;

  return (
    <Card className="card-elevated">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-accent" />
          Concept Analysis Visualization
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Overview Chart */}
        <div className="space-y-4">
          <h3 className="font-semibold flex items-center gap-2">
            <PieChart className="w-4 h-4" />
            Concept Distribution
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Matches</span>
                <span className="text-sm text-muted-foreground">{matching.length}</span>
              </div>
              <Progress value={matchPercentage} className="h-2 bg-success-light">
                <div className="h-full bg-success rounded-full transition-all" 
                     style={{width: `${matchPercentage}%`}} />
              </Progress>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Conflicts</span>
                <span className="text-sm text-muted-foreground">{conflicting.length}</span>
              </div>
              <Progress value={conflictPercentage} className="h-2 bg-destructive-light">
                <div className="h-full bg-destructive rounded-full transition-all" 
                     style={{width: `${conflictPercentage}%`}} />
              </Progress>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Fidelity</span>
                <span className="text-sm text-muted-foreground">{(fidelityScore * 100).toFixed(1)}%</span>
              </div>
              <Progress value={fidelityScore * 100} className="h-2" />
            </div>
          </div>
        </div>

        {/* Detailed Breakdown */}
        <div className="space-y-4">
          <h3 className="font-semibold flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Detailed Analysis
          </h3>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-success">✓ Matching Concepts ({matching.length})</h4>
                <div className="space-y-1">
                  {matching.map((concept, index) => (
                    <div key={index} className="flex items-center gap-2 p-2 bg-success-light rounded-md">
                      <div className="w-2 h-2 bg-success rounded-full flex-shrink-0"></div>
                      <span className="text-sm">{concept}</span>
                    </div>
                  ))}
                  {matching.length === 0 && (
                    <p className="text-sm text-muted-foreground italic">No matching concepts found</p>
                  )}
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="text-sm font-medium text-destructive">✗ Conflicting Concepts ({conflicting.length})</h4>
                <div className="space-y-1">
                  {conflicting.map((concept, index) => (
                    <div key={index} className="flex items-center gap-2 p-2 bg-destructive-light rounded-md">
                      <div className="w-2 h-2 bg-destructive rounded-full flex-shrink-0"></div>
                      <span className="text-sm">{concept}</span>
                    </div>
                  ))}
                  {conflicting.length === 0 && (
                    <p className="text-sm text-muted-foreground italic">No conflicting concepts found</p>
                  )}
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="space-y-3">
                <h4 className="text-sm font-medium text-muted-foreground">Hypothesis Concepts ({hypothesisConcepts.length})</h4>
                <div className="flex flex-wrap gap-2">
                  {hypothesisConcepts.map((concept, index) => {
                    const isMatching = matching.includes(concept);
                    const isConflicting = conflicting.includes(concept);
                    return (
                      <Badge 
                        key={index} 
                        variant={isMatching ? "success" : isConflicting ? "destructive" : "outline"}
                        className="text-xs"
                      >
                        {concept}
                      </Badge>
                    );
                  })}
                </div>
              </div>

              <div className="space-y-3">
                <h4 className="text-sm font-medium text-muted-foreground">Reference Concepts ({referenceConcepts.length})</h4>
                <div className="flex flex-wrap gap-2">
                  {referenceConcepts.map((concept, index) => (
                    <Badge key={index} variant="secondary" className="text-xs">
                      {concept}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Summary */}
        <div className="p-4 bg-muted/30 rounded-lg border border-border">
          <div className="flex items-start gap-3">
            <TrendingUp className="w-5 h-5 text-accent mt-0.5" />
            <div className="space-y-1">
              <h4 className="font-medium">Analysis Summary</h4>
              <p className="text-sm text-muted-foreground">
                {matching.length > conflicting.length 
                  ? `Strong alignment detected with ${matching.length} matching concepts and ${conflicting.length} conflicts.`
                  : conflicting.length > matching.length
                  ? `Significant misalignment with ${conflicting.length} conflicts versus ${matching.length} matches.`
                  : `Balanced result with equal matching and conflicting concepts.`
                }
                {fidelityScore > 0.7 ? " High fidelity score indicates good overall alignment." :
                 fidelityScore > 0.4 ? " Moderate fidelity suggests room for improvement." :
                 " Low fidelity indicates significant discrepancies."}
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default ConceptVisualization;