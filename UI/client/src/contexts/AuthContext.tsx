import React, { createContext, useContext, useState, useCallback } from "react";

interface User {
  name: string;
  email: string;
  avatar?: string;
  faceRegistered: boolean;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isOnboarding: boolean;
  login: (user: User) => void;
  logout: () => void;
  setIsOnboarding: (val: boolean) => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const STORAGE_KEY = "rag_user";

function loadUser(): User | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as User) : null;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(loadUser);
  const [isOnboarding, setIsOnboarding] = useState(false);

  const login = useCallback((userData: User) => {
    setUser(userData);
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(userData));
    } catch {
      /* ignore storage errors */
    }
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch {
      /* ignore storage errors */
    }
  }, []);

  const isAuthenticated = user !== null;

  return (
    <AuthContext.Provider value={{ user, isAuthenticated, isOnboarding, login, logout, setIsOnboarding }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
