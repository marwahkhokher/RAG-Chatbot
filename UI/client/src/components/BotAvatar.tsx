import { motion } from "framer-motion";
import { useState, useEffect } from "react";

interface BotAvatarProps {
  size?: "sm" | "md" | "lg" | "xl";
  className?: string;
  showGreeting?: boolean;
}

const sizeMap = {
  sm: 36,
  md: 56,
  lg: 96,
  xl: 140,
};

const greetingTexts = [
  "Hello there!",
  "Ready to explore?",
  "Let's discover together.",
  "I'm here to help.",
];

export function BotAvatar({ size = "md", className = "", showGreeting = false }: BotAvatarProps) {
  const pixelSize = sizeMap[size];
  const [greetingIndex, setGreetingIndex] = useState(0);

  useEffect(() => {
    if (!showGreeting) return;
    const interval = setInterval(() => {
      setGreetingIndex((i) => (i + 1) % greetingTexts.length);
    }, 3000);
    return () => clearInterval(interval);
  }, [showGreeting]);

  return (
    <div className={`relative flex items-center justify-center ${className}`}>
      {/* Outer glow ring */}
      <motion.div
        className="absolute inset-0 rounded-full"
        animate={{
          boxShadow: [
            "0 0 20px oklch(0.82 0.15 175 / 0.2), 0 0 40px oklch(0.82 0.15 175 / 0.1)",
            "0 0 30px oklch(0.82 0.15 175 / 0.35), 0 0 60px oklch(0.82 0.15 175 / 0.15)",
            "0 0 20px oklch(0.82 0.15 175 / 0.2), 0 0 40px oklch(0.82 0.15 175 / 0.1)",
          ],
        }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        style={{ width: pixelSize, height: pixelSize }}
      />

      {/* Pulsing ring */}
      <motion.div
        className="absolute rounded-full border border-teal/20"
        animate={{ scale: [0.9, 1.3], opacity: [0.3, 0] }}
        transition={{ duration: 3, repeat: Infinity, ease: "easeOut" }}
        style={{ width: pixelSize, height: pixelSize }}
      />

      {/* Neural-node avatar (CSS/SVG — no external asset needed) */}
      <motion.div
        className="relative z-10 rounded-full flex items-center justify-center"
        style={{
          width: pixelSize,
          height: pixelSize,
          background:
            "radial-gradient(circle at 35% 30%, oklch(0.82 0.15 175 / 0.30), oklch(0.62 0.15 275 / 0.16))",
          border: "1px solid oklch(0.82 0.15 175 / 0.4)",
          boxShadow: "inset 0 0 16px oklch(0.82 0.15 175 / 0.18)",
        }}
        animate={{ scale: [0.98, 1.02, 0.98] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        whileHover={{ scale: 1.08 }}
      >
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="oklch(0.82 0.15 175)"
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ width: pixelSize * 0.46, height: pixelSize * 0.46 }}
        >
          <circle cx="12" cy="5.5" r="2.3" fill="oklch(0.82 0.15 175 / 0.9)" stroke="none" />
          <circle cx="5.5" cy="16" r="2.3" fill="oklch(0.62 0.15 275 / 0.9)" stroke="none" />
          <circle cx="18.5" cy="16" r="2.3" fill="oklch(0.82 0.15 175 / 0.9)" stroke="none" />
          <path d="M12 7.8 L6.6 13.9 M12 7.8 L17.4 13.9 M7.6 16 L16.4 16" />
        </svg>
      </motion.div>

      {/* Greeting tooltip */}
      {showGreeting && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          key={greetingIndex}
          className="absolute -right-28 bg-white/[0.06] backdrop-blur-xl border border-white/[0.08] rounded-xl px-3 py-1.5 text-xs text-foreground/80 whitespace-nowrap"
          style={{ fontFamily: "var(--font-sans)" }}
        >
          {greetingTexts[greetingIndex]}
          <div className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 bg-white/[0.06] border-l border-t border-white/[0.08] rotate-45" />
        </motion.div>
      )}

      {/* Status indicator */}
      <div className="absolute bottom-0 right-0 z-20 w-3 h-3 rounded-full bg-teal shadow-[0_0_8px_oklch(0.82_0.15_175_/0.6)]" />
    </div>
  );
}
