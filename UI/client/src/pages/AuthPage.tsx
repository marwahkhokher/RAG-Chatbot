/*
 * Ethereal Glass Design — Auth Page
 * Wired to the RAG-Chatbot FastAPI backend: real facial register/login via
 * /auth/register and /auth/login. Face-only (username + face) — no passwords.
 */
import { useState, useRef, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BotAvatar } from "@/components/BotAvatar";
import { ParticleBackground } from "@/components/ParticleBackground";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { useLocation } from "wouter";
import { toast } from "sonner";
import {
  User, ChevronRight, Camera, Scan, CheckCircle2, Sparkles, ArrowLeft, Shield,
} from "lucide-react";
import { registerFace, loginFace } from "@/lib/api";

type AuthView = "welcome" | "signup" | "signin" | "face-scan" | "success";

const onboardingMessages = [
  "Welcome! I'm your AI companion.",
  "I'll set up facial recognition for secure access.",
  "It's quick and keeps your conversations private.",
  "Ready? Let's capture your face.",
];

const containerVariants = {
  initial: { opacity: 0 },
  animate: { opacity: 1, transition: { duration: 0.5, ease: [0.23, 1, 0.32, 1] as any } },
  exit: { opacity: 0, transition: { duration: 0.3 } },
};

const slideVariants = {
  initial: { opacity: 0, x: 30, y: 10 },
  animate: { opacity: 1, x: 0, y: 0, transition: { duration: 0.5, ease: [0.34, 1.56, 0.64, 1] as any } },
  exit: { opacity: 0, x: -30, y: 10, transition: { duration: 0.3 } },
};

export default function AuthPage() {
  const [view, setView] = useState<AuthView>("welcome");
  const [authMode, setAuthMode] = useState<"signup" | "signin">("signup");
  const [name, setName] = useState("");
  const [onboardingStep, setOnboardingStep] = useState(0);
  const [scanning, setScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { login } = useAuth();
  const [, navigate] = useLocation();

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 320, height: 320 },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
    } catch {
      // Graceful fallback — no camera available
    }
  }, []);

  const stopCamera = useCallback(() => {
    const stream = videoRef.current?.srcObject as MediaStream | null;
    stream?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  // Warm up the camera when entering the face-scan view.
  useEffect(() => {
    if (view === "face-scan") startCamera();
    return () => {
      if (view === "face-scan") stopCamera();
    };
  }, [view, startCamera, stopCamera]);

  const captureFrame = useCallback((): string => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return "";
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.9);
  }, []);

  const startFaceScan = useCallback(async () => {
    setScanning(true);
    setScanProgress(0);

    const interval = setInterval(() => {
      setScanProgress((p) => Math.min(p + 3, 100));
    }, 60);

    // Let the animation run and the camera settle, then capture.
    await new Promise((r) => setTimeout(r, 2200));
    clearInterval(interval);
    setScanProgress(100);

    const image = captureFrame();
    stopCamera();
    setScanning(false);

    if (!image) {
      toast.error("Couldn't access the camera. Try again, or continue as guest.");
      setScanProgress(0);
      return;
    }

    try {
      if (authMode === "signup") {
        const res = await registerFace(name.trim(), image);
        if (res.authenticated) {
          setView("success");
        } else {
          toast.error(res.message || "Registration failed. Try again.");
          setScanProgress(0);
        }
      } else {
        const res = await loginFace(image);
        if (res.authenticated) {
          login({ name: res.username, email: "", faceRegistered: true });
          navigate("/chat");
        } else {
          toast.error(res.message || "Face not recognized. Try again or register.");
          setScanProgress(0);
        }
      }
    } catch (err: any) {
      toast.error(err?.response?.data?.detail || "Server error. Is the backend running?");
      setScanProgress(0);
    }
  }, [authMode, name, login, navigate, captureFrame, stopCamera]);

  useEffect(() => {
    if (view === "welcome" && onboardingStep < onboardingMessages.length) {
      const timer = setTimeout(() => setOnboardingStep((prev) => prev + 1), 2500);
      return () => clearTimeout(timer);
    }
  }, [view, onboardingStep]);

  const handleContinueSignup = () => {
    if (!name.trim()) {
      toast.error("Please enter your name");
      return;
    }
    setView("face-scan");
    setScanning(false);
    setScanProgress(0);
  };

  const continueAsGuest = () => {
    login({ name: "Guest", email: "", faceRegistered: false });
    navigate("/chat");
  };

  const primaryBtn = {
    background: "linear-gradient(135deg, oklch(0.82 0.15 175), oklch(0.72 0.14 180))",
    color: "oklch(0.12 0.02 270)",
    boxShadow: "0 0 20px oklch(0.82 0.15 175 / 0.2)",
  };

  return (
    <div className="min-h-screen bg-background relative overflow-hidden flex items-center justify-center">
      <div className="fixed inset-0 bg-gradient-to-b from-background/40 via-background/60 to-background/90" />
      <ParticleBackground count={15} />

      <AnimatePresence mode="wait">
        <motion.div
          key={view}
          variants={containerVariants}
          initial="initial"
          animate="animate"
          exit="exit"
          className="relative z-10 w-full max-w-5xl mx-auto px-4 py-8"
        >
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
            {/* Left: Bot Assistant */}
            <motion.div
              variants={slideVariants}
              className="flex flex-col items-center justify-center text-center space-y-6 lg:pr-8"
            >
              <BotAvatar size="xl" showGreeting={view === "welcome"} />
              <motion.div
                className="space-y-3"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.5 }}
              >
                <h1 className="text-3xl lg:text-4xl font-bold text-glow-teal" style={{ color: "var(--color-teal)" }}>
                  RAG Chatbot
                </h1>
                <p className="text-sm text-muted-foreground max-w-xs mx-auto leading-relaxed">
                  Your AI-powered knowledge companion. Ask anything, grounded in your data.
                </p>
              </motion.div>

              {view === "welcome" && (
                <div className="space-y-2 mt-4 min-h-[80px]">
                  <AnimatePresence mode="wait">
                    <motion.p
                      key={onboardingStep}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      transition={{ duration: 0.4 }}
                      className="text-sm text-foreground/70 italic max-w-xs mx-auto"
                    >
                      "{onboardingMessages[Math.min(onboardingStep, onboardingMessages.length - 1)]}"
                    </motion.p>
                  </AnimatePresence>
                </div>
              )}
            </motion.div>

            {/* Right: Auth Forms */}
            <motion.div variants={slideVariants} className="w-full max-w-md mx-auto lg:ml-auto">
              <AnimatePresence mode="wait">
                {view === "welcome" && (
                  <motion.div
                    key="welcome"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="glass rounded-2xl p-8 space-y-6"
                  >
                    <div className="space-y-2">
                      <h2 className="text-xl font-semibold text-foreground" style={{ fontFamily: "var(--font-display)" }}>
                        Begin your conversation
                      </h2>
                      <p className="text-sm text-muted-foreground">
                        Register your face or sign in to get started.
                      </p>
                    </div>

                    <div className="space-y-3">
                      <Button
                        onClick={() => {
                          setAuthMode("signup");
                          setView("signup");
                        }}
                        className="w-full h-12 text-base font-medium active:scale-[0.97]"
                        style={primaryBtn}
                      >
                        <Sparkles className="w-4 h-4 mr-2" />
                        Create Account
                      </Button>

                      <Button
                        onClick={() => {
                          setAuthMode("signin");
                          setView("face-scan");
                        }}
                        variant="outline"
                        className="w-full h-12 text-base font-medium glass glass-hover active:scale-[0.97]"
                        style={{ borderColor: "oklch(1 0 0 / 0.12)", color: "var(--color-teal)" }}
                      >
                        Sign In with Face
                      </Button>

                      <button
                        onClick={continueAsGuest}
                        className="w-full text-xs text-muted-foreground hover:text-foreground transition-colors pt-1"
                      >
                        Continue as guest
                      </button>
                    </div>

                    <div className="flex items-center gap-3 text-xs text-muted-foreground">
                      <Shield className="w-3 h-3" />
                      <span>Secured with facial recognition</span>
                    </div>
                  </motion.div>
                )}

                {view === "signup" && (
                  <motion.div
                    key="signup"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    className="glass rounded-2xl p-8 space-y-6"
                  >
                    <button
                      onClick={() => setView("welcome")}
                      className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <ArrowLeft className="w-3 h-3" />
                      Back
                    </button>

                    <div className="space-y-1">
                      <h2 className="text-xl font-semibold text-foreground" style={{ fontFamily: "var(--font-display)" }}>
                        Create your account
                      </h2>
                      <p className="text-sm text-muted-foreground">
                        Enter your name, then register your face for secure access.
                      </p>
                    </div>

                    <div className="relative group">
                      <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-teal transition-colors" />
                      <input
                        type="text"
                        placeholder="Your name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleContinueSignup()}
                        className="w-full h-11 pl-10 pr-4 rounded-xl bg-white/[0.04] border border-white/[0.06] text-foreground text-sm placeholder:text-muted-foreground/50 focus:outline-none focus:border-teal/40 focus:bg-white/[0.06] transition-all duration-200"
                      />
                    </div>

                    <Button onClick={handleContinueSignup} className="w-full h-12 text-base font-medium active:scale-[0.97]" style={primaryBtn}>
                      Continue to Face Setup
                      <ChevronRight className="w-4 h-4 ml-2" />
                    </Button>

                    <p className="text-center text-xs text-muted-foreground">
                      Already registered?{" "}
                      <button
                        onClick={() => {
                          setAuthMode("signin");
                          setView("face-scan");
                        }}
                        className="text-teal hover:underline"
                      >
                        Sign in
                      </button>
                    </p>
                  </motion.div>
                )}

                {view === "face-scan" && (
                  <motion.div
                    key="face-scan"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="glass rounded-2xl p-8 space-y-6"
                  >
                    <button
                      onClick={() => {
                        stopCamera();
                        setView(authMode === "signup" ? "signup" : "welcome");
                      }}
                      className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                    >
                      <ArrowLeft className="w-3 h-3" />
                      Back
                    </button>

                    <div className="space-y-2 text-center">
                      <h2 className="text-xl font-semibold text-foreground" style={{ fontFamily: "var(--font-display)" }}>
                        {authMode === "signup" ? "Register your face" : "Sign in with your face"}
                      </h2>
                      <p className="text-sm text-muted-foreground">
                        {scanning ? "Scanning your face..." : "Position your face in the frame and tap scan."}
                      </p>
                    </div>

                    <div className="relative w-56 h-56 mx-auto rounded-2xl overflow-hidden bg-black/40 border border-white/[0.06]">
                      <video ref={videoRef} className="w-full h-full object-cover rounded-2xl" muted playsInline />
                      <canvas ref={canvasRef} className="hidden" />

                      {scanning && (
                        <>
                          <div className="absolute inset-0 grid grid-cols-3 grid-rows-3 gap-px p-6">
                            {Array.from({ length: 9 }).map((_, i) => (
                              <div key={i} className="border border-teal/20 rounded-sm" />
                            ))}
                          </div>
                          <motion.div
                            className="absolute left-4 right-4 h-[2px] bg-gradient-to-r from-transparent via-teal to-transparent"
                            animate={{ top: ["8%", "92%", "8%"] }}
                            transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
                          />
                          <div className="absolute top-3 left-3 w-8 h-8 border-l-2 border-t-2 border-teal rounded-tl-lg" />
                          <div className="absolute top-3 right-3 w-8 h-8 border-r-2 border-t-2 border-teal rounded-tr-lg" />
                          <div className="absolute bottom-3 left-3 w-8 h-8 border-l-2 border-b-2 border-teal rounded-bl-lg" />
                          <div className="absolute bottom-3 right-3 w-8 h-8 border-r-2 border-b-2 border-teal rounded-br-lg" />
                        </>
                      )}
                    </div>

                    {scanning && (
                      <div className="w-full h-1 bg-white/[0.06] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-teal to-violet rounded-full transition-all"
                          style={{ width: `${scanProgress}%` }}
                        />
                      </div>
                    )}

                    {!scanning && (
                      <Button
                        onClick={startFaceScan}
                        className="w-full h-12 text-base font-medium active:scale-[0.97]"
                        style={{
                          background: "linear-gradient(135deg, oklch(0.82 0.15 175), oklch(0.62 0.15 275))",
                          color: "oklch(0.95 0 0)",
                          boxShadow: "0 0 20px oklch(0.82 0.15 175 / 0.2)",
                        }}
                      >
                        <Scan className="w-4 h-4 mr-2" />
                        Start Face Scan
                      </Button>
                    )}

                    {scanning && (
                      <div className="flex items-center justify-center gap-2 text-sm text-teal">
                        <motion.div
                          className="w-2 h-2 rounded-full bg-teal"
                          animate={{ opacity: [0.3, 1, 0.3] }}
                          transition={{ duration: 1, repeat: Infinity }}
                        />
                        <span>Analyzing facial features...</span>
                      </div>
                    )}

                    {!scanning && (
                      <div className="flex items-center justify-center gap-1.5 text-xs text-muted-foreground/60">
                        <Camera className="w-3 h-3" />
                        <span>Your camera activates when you tap scan.</span>
                      </div>
                    )}
                  </motion.div>
                )}

                {view === "success" && (
                  <motion.div
                    key="success"
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="glass rounded-2xl p-8 space-y-6 text-center"
                  >
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 200, damping: 15 }}
                      className="w-20 h-20 mx-auto rounded-full bg-teal/20 flex items-center justify-center"
                    >
                      <CheckCircle2 className="w-10 h-10 text-teal" />
                    </motion.div>

                    <div className="space-y-2">
                      <h2 className="text-xl font-semibold text-foreground" style={{ fontFamily: "var(--font-display)" }}>
                        You're all set, {name || "friend"}!
                      </h2>
                      <p className="text-sm text-muted-foreground">
                        Facial recognition registered. Welcome to RAG Chatbot.
                      </p>
                    </div>

                    <BotAvatar size="md" className="mx-auto" />

                    <Button
                      onClick={() => {
                        login({ name: name.trim() || "User", email: "", faceRegistered: true });
                        navigate("/chat");
                      }}
                      className="w-full h-12 text-base font-medium active:scale-[0.97]"
                      style={primaryBtn}
                    >
                      Enter Workspace
                      <ChevronRight className="w-4 h-4 ml-2" />
                    </Button>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
