import { useState } from "react";
import EvaluationHeader from "@/components/evaluation/EvaluationHeader";
import UserInputSection from "@/components/evaluation/UserInputSection";
import TruthReferenceSection from "@/components/evaluation/TruthReferenceSection";
import EvaluationMetrics from "@/components/evaluation/EvaluationMetrics";
import ConceptVisualization from "@/components/evaluation/ConceptVisualization";
import { useToast } from "@/hooks/use-toast";

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

const EvaluationInterface = () => {
  const [goldAnswer, setGoldAnswer] = useState("");
  const [evaluationResult, setEvaluationResult] = useState<EvaluationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  // Simulated evaluation logic (in a real app, this would call an API)
  const runEvaluation = async (prompt: string, llmOutput: string) => {
    if (!goldAnswer.trim()) {
      toast({
        title: "Missing Reference",
        description: "Please provide a gold/reference answer before running evaluation.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 2000));

    try {
      // Simple concept extraction (in reality, this would use NLP/AI)
      const extractConcepts = (text: string): string[] => {
        return text.toLowerCase()
          .replace(/[^\w\s]/g, ' ')
          .split(/\s+/)
          .filter(word => word.length > 2)
          .slice(0, 10); // Limit for demo
      };

      const hypothesisConcepts = extractConcepts(llmOutput);
      const referenceConcepts = extractConcepts(goldAnswer);
      
      // Find matches and conflicts
      const matchingConcepts = hypothesisConcepts.filter(concept => 
        referenceConcepts.includes(concept)
      );
      
      const conflictingConcepts = hypothesisConcepts.filter(concept => 
        !referenceConcepts.includes(concept) && concept.length > 3
      );

      // Calculate fidelity score
      const totalConcepts = Math.max(hypothesisConcepts.length, referenceConcepts.length);
      const fidelityScore = totalConcepts > 0 ? matchingConcepts.length / totalConcepts : 0;

      // Determine hallucination risk
      const conflictRatio = conflictingConcepts.length / Math.max(hypothesisConcepts.length, 1);
      let hallucinationRisk: "Low" | "Medium" | "High" = "Low";
      if (conflictRatio > 0.6) hallucinationRisk = "High";
      else if (conflictRatio > 0.3) hallucinationRisk = "Medium";

      const result: EvaluationResult = {
        fidelity_score: fidelityScore,
        hallucination_risk: hallucinationRisk,
        analysis: {
          hypothesis_concepts: hypothesisConcepts,
          reference_concepts: referenceConcepts,
          matching_concepts: matchingConcepts,
          conflicting_concepts: conflictingConcepts,
        },
        rationale: `The hypothesis ${matchingConcepts.length > 0 ? `correctly identifies ${matchingConcepts.join(', ')}` : 'shares no common concepts with the reference'} ${conflictingConcepts.length > 0 ? `but introduces conflicting concepts: ${conflictingConcepts.join(', ')}.` : '.'} Fidelity score of ${(fidelityScore * 100).toFixed(1)}% indicates ${fidelityScore > 0.7 ? 'strong' : fidelityScore > 0.4 ? 'moderate' : 'weak'} alignment.`
      };

      setEvaluationResult(result);
      
      toast({
        title: "Evaluation Complete",
        description: `Analysis finished with ${result.hallucination_risk.toLowerCase()} hallucination risk.`,
        variant: result.hallucination_risk === "High" ? "destructive" : "default",
      });

    } catch (error) {
      toast({
        title: "Evaluation Failed",
        description: "An error occurred during evaluation. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <EvaluationHeader />
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <UserInputSection onEvaluate={runEvaluation} isLoading={isLoading} />
          <TruthReferenceSection goldAnswer={goldAnswer} onGoldAnswerChange={setGoldAnswer} />
        </div>

        {evaluationResult && (
          <div className="space-y-6">
            <EvaluationMetrics result={evaluationResult} />
            <ConceptVisualization
              matching={evaluationResult.analysis.matching_concepts}
              conflicting={evaluationResult.analysis.conflicting_concepts}
              hypothesisConcepts={evaluationResult.analysis.hypothesis_concepts}
              referenceConcepts={evaluationResult.analysis.reference_concepts}
              fidelityScore={evaluationResult.fidelity_score}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default EvaluationInterface;