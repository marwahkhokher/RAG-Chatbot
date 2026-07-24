import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import AuthPage from "./pages/AuthPage";
import ChatPage from "./pages/ChatPage";
import { AuthProvider } from "./contexts/AuthContext";

function Router() {
  return (
    <Switch>
      <Route path={"/"} component={AuthPage} />
      <Route path={"/auth"} component={AuthPage} />
      <Route path={"/chat"} component={ChatPage} />
      <Route path={"/404"} component={NotFound} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="dark">
        <AuthProvider>
          <TooltipProvider>
            <Toaster
              position="top-center"
              toastOptions={{
                style: {
                  background: 'oklch(0.15 0.02 270 / 0.9)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid oklch(1 0 0 / 0.08)',
                  color: 'oklch(0.9 0.01 270)',
                  borderRadius: '0.75rem',
                  fontFamily: 'var(--font-sans)',
                }
              }}
            />
            <Router />
          </TooltipProvider>
        </AuthProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
