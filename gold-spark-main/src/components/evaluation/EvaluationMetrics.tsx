import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { BarChart3, AlertTriangle, CheckCircle, XCircle, Download, Eye } from "lucide-react";
import { useState } from "react";

interface EvaluationResult {
  fidelity_score: number;
  hallucination_risk: "Low" | "Medium" | "High";
  analysis: {
    hypothesis_concepts: string[];
    reference_concepts: string[];
    matching_concepts: string[];
    conflicting_concepts: string[];
  };
  rationale: string;
}

interface EvaluationMetricsProps {
  result: EvaluationResult | null;
}

const EvaluationMetrics = ({ result }: EvaluationMetricsProps) => {
  const [showJsonView, setShowJsonView] = useState(false);

  if (!result) {
    return (
      <Card className="card-elevated">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-accent" />
            Evaluation Metrics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>Run an evaluation to see metrics and analysis</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "Low": return "success";
      case "Medium": return "warning";
      case "High": return "destructive";
      default: return "secondary";
    }
  };

  const getRiskIcon = (risk: string) => {
    switch (risk) {
      case "Low": return <CheckCircle className="w-4 h-4" />;
      case "Medium": return <AlertTriangle className="w-4 h-4" />;
      case "High": return <XCircle className="w-4 h-4" />;
      default: return <AlertTriangle className="w-4 h-4" />;
    }
  };

  return (
    <Card className="card-elevated">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-accent" />
            Evaluation Results
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => setShowJsonView(!showJsonView)}
            >
              <Eye className="w-4 h-4" />
              {showJsonView ? "Visual" : "JSON"}
            </Button>
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4" />
              Export
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {showJsonView ? (
          <div className="code-block">
            <pre className="text-sm whitespace-pre-wrap">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        ) : (
          <>
            {/* Scores Section */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <h3 className="font-semibold">Fidelity Score</h3>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Similarity</span>
                    <span className="font-mono font-medium">{result.fidelity_score.toFixed(4)}</span>
                  </div>
                  <Progress value={result.fidelity_score * 100} className="h-2" />
                </div>
              </div>

              <div className="space-y-3">
                <h3 className="font-semibold">Hallucination Risk</h3>
                <Badge variant={getRiskColor(result.hallucination_risk)} className="flex items-center gap-2 w-fit">
                  {getRiskIcon(result.hallucination_risk)}
                  {result.hallucination_risk} Risk
                </Badge>
              </div>
            </div>

            {/* Concept Analysis */}
            <div className="space-y-4">
              <h3 className="font-semibold">Concept Analysis</h3>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-muted-foreground">Hypothesis Concepts</h4>
                    <div className="flex flex-wrap gap-2">
                      {result.analysis.hypothesis_concepts.map((concept, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {concept}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-muted-foreground">Reference Concepts</h4>
                    <div className="flex flex-wrap gap-2">
                      {result.analysis.reference_concepts.map((concept, index) => (
                        <Badge key={index} variant="secondary" className="text-xs">
                          {concept}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-success" />
                      Matching Concepts
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {result.analysis.matching_concepts.map((concept, index) => (
                        <Badge key={index} className="status-match text-xs">
                          {concept}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h4 className="text-sm font-medium flex items-center gap-2">
                      <XCircle className="w-4 h-4 text-destructive" />
                      Conflicting Concepts
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {result.analysis.conflicting_concepts.map((concept, index) => (
                        <Badge key={index} className="status-conflict text-xs">
                          {concept}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Rationale */}
            <div className="space-y-2">
              <h3 className="font-semibold">Analysis Rationale</h3>
              <div className="p-4 bg-muted/50 rounded-lg border border-border">
                <p className="text-sm">{result.rationale}</p>
              </div>
            </div>

            {/* Summary Stats */}
            <div className="grid grid-cols-3 gap-4 pt-4 border-t border-border">
              <div className="text-center">
                <div className="text-2xl font-bold text-success">{result.analysis.matching_concepts.length}</div>
                <div className="text-xs text-muted-foreground">Matches</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-destructive">{result.analysis.conflicting_concepts.length}</div>
                <div className="text-xs text-muted-foreground">Conflicts</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-accent">{(result.fidelity_score * 100).toFixed(1)}%</div>
                <div className="text-xs text-muted-foreground">Fidelity</div>
              </div>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default EvaluationMetrics;