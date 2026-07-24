import { motion } from "framer-motion";
import { FileText, Brain, Lightbulb, Search } from "lucide-react";

interface SuggestedPromptsProps {
  onSelect: (prompt: string) => void;
}

const prompts = [
  {
    icon: Brain,
    title: "Analyze my documents",
    subtitle: "Find patterns and insights in your uploaded files",
    color: "oklch(0.82 0.15 175)",
  },
  {
    icon: Search,
    title: "Search my knowledge base",
    subtitle: "Ask questions grounded in your data",
    color: "oklch(0.62 0.15 275)",
  },
  {
    icon: Lightbulb,
    title: "Generate insights",
    subtitle: "Get AI-powered recommendations and summaries",
    color: "oklch(0.72 0.18 15)",
  },
  {
    icon: FileText,
    title: "Create a report",
    subtitle: "Automatically generate structured documents",
    color: "oklch(0.5 0.1 200)",
  },
];

const containerVariants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.08,
      delayChildren: 0.2,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.34, 1.56, 0.64, 1] as any } },
};

export function SuggestedPrompts({ onSelect }: SuggestedPromptsProps) {
  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-2xl mx-auto"
    >
      {prompts.map((prompt, i) => {
        const Icon = prompt.icon;
        return (
          <motion.button
            key={i}
            variants={itemVariants}
            onClick={() => onSelect(prompt.title)}
            className="glass glass-hover rounded-xl p-4 text-left transition-all duration-200 active:scale-[0.97] group"
            style={{
              borderColor: "oklch(1 0 0 / 0.06)",
            }}
          >
            <div className="flex items-start gap-3">
              <div
                className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 transition-all duration-200"
                style={{
                  background: `${prompt.color}15`,
                  border: `1px solid ${prompt.color}30`,
                }}
              >
                <Icon className="w-4 h-4" style={{ color: prompt.color }} />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground group-hover:text-teal transition-colors duration-200" style={{ fontFamily: "var(--font-sans)" }}>
                  {prompt.title}
                </p>
                <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">
                  {prompt.subtitle}
                </p>
              </div>
            </div>
          </motion.button>
        );
      })}
    </motion.div>
  );
}
