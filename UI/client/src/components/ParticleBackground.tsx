import { useMemo } from "react";

interface Particle {
  id: number;
  left: string;
  size: number;
  duration: string;
  delay: string;
  color: string;
}

export function ParticleBackground({ count = 20 }: { count?: number }) {
  const particles = useMemo(() => {
    return Array.from({ length: count }, (_, i) => ({
      id: i,
      left: `${Math.random() * 100}%`,
      size: Math.random() * 4 + 1,
      duration: `${Math.random() * 15 + 10}s`,
      delay: `${Math.random() * 10}s`,
      color: Math.random() > 0.5 ? "oklch(0.82 0.15 175 / 0.4)" : "oklch(0.62 0.15 275 / 0.3)",
    }));
  }, [count]);

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
      {particles.map((p) => (
        <div
          key={p.id}
          className="particle"
          style={{
            left: p.left,
            width: `${p.size}px`,
            height: `${p.size}px`,
            background: p.color,
            animationDuration: p.duration,
            animationDelay: p.delay,
            bottom: "-10px",
          }}
        />
      ))}
    </div>
  );
}
