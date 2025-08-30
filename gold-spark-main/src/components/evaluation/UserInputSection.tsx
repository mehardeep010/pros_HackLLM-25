import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Send, RefreshCw } from "lucide-react";
import { useState } from "react";

interface UserInputSectionProps {
  onEvaluate: (prompt: string, llmOutput: string) => void;
  isLoading?: boolean;
}

const UserInputSection = ({ onEvaluate, isLoading = false }: UserInputSectionProps) => {
  const [prompt, setPrompt] = useState("");
  const [llmOutput, setLlmOutput] = useState("");

  const handleEvaluate = () => {
    if (prompt.trim() && llmOutput.trim()) {
      onEvaluate(prompt, llmOutput);
    }
  };

  const handleLoadExample = () => {
    setPrompt("Where is the Eiffel Tower located?");
    setLlmOutput("The Eiffel Tower is in Rome, Italy.");
  };

  return (
    <Card className="card-elevated">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Send className="w-5 h-5 text-accent" />
            User Input Section
          </CardTitle>
          <Button variant="outline" size="sm" onClick={handleLoadExample}>
            <RefreshCw className="w-4 h-4" />
            Load Example
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <label htmlFor="prompt" className="text-sm font-medium">
            Prompt (User's Question/Instruction)
          </label>
          <Textarea
            id="prompt"
            placeholder="Enter your prompt here..."
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="min-h-[100px] focus-enhanced"
            rows={4}
          />
        </div>
        
        <div className="space-y-2">
          <label htmlFor="llm-output" className="text-sm font-medium">
            LLM Output (Generated Response)
          </label>
          <Textarea
            id="llm-output"
            placeholder="Enter the model's output here..."
            value={llmOutput}
            onChange={(e) => setLlmOutput(e.target.value)}
            className="min-h-[120px] focus-enhanced"
            rows={5}
          />
        </div>

        <Button 
          variant="evaluation" 
          className="w-full" 
          onClick={handleEvaluate}
          disabled={!prompt.trim() || !llmOutput.trim() || isLoading}
        >
          {isLoading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              Evaluating...
            </>
          ) : (
            <>
              <Send className="w-4 h-4" />
              Run Evaluation
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );
};

export default UserInputSection;