import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Target, Download, Upload } from "lucide-react";
import { useState } from "react";

interface TruthReferenceSectionProps {
  goldAnswer: string;
  onGoldAnswerChange: (answer: string) => void;
}

const TruthReferenceSection = ({ goldAnswer, onGoldAnswerChange }: TruthReferenceSectionProps) => {
  const [isJsonView, setIsJsonView] = useState(false);

  const handleLoadExample = () => {
    onGoldAnswerChange("The Eiffel Tower is in Paris, France, located on the Champ de Mars.");
  };

  const formatAsJson = (text: string) => {
    return JSON.stringify({ gold_answer: text }, null, 2);
  };

  return (
    <Card className="card-elevated">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Target className="w-5 h-5 text-success" />
            Truth/Gold Reference Section
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => setIsJsonView(!isJsonView)}
            >
              {isJsonView ? "Text View" : "JSON View"}
            </Button>
            <Button variant="outline" size="sm" onClick={handleLoadExample}>
              <Upload className="w-4 h-4" />
              Load Example
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label htmlFor="gold-answer" className="text-sm font-medium">
            Ground Truth Answer
          </label>
          {isJsonView ? (
            <div className="code-block">
              <pre className="text-sm whitespace-pre-wrap">
                {formatAsJson(goldAnswer)}
              </pre>
            </div>
          ) : (
            <Textarea
              id="gold-answer"
              placeholder="Enter the correct/reference answer here..."
              value={goldAnswer}
              onChange={(e) => onGoldAnswerChange(e.target.value)}
              className="min-h-[120px] focus-enhanced"
              rows={5}
            />
          )}
        </div>

        {goldAnswer && (
          <div className="flex items-center gap-2 pt-2">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <div className="w-2 h-2 rounded-full bg-success"></div>
              Reference loaded ({goldAnswer.split(' ').length} words)
            </div>
            <Button variant="ghost" size="sm">
              <Download className="w-4 h-4" />
              Export
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default TruthReferenceSection;