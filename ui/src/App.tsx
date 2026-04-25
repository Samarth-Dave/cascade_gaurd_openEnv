import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppShell } from "@/components/layout/AppShell";
import { CascadeProvider } from "@/context/CascadeContext";
import AnalyticsPage from "./pages/AnalyticsPage";
import AboutPage from "./pages/AboutPage";
import DatasetPage from "./pages/DatasetPage";
import HistoryPage from "./pages/HistoryPage";
import NotFound from "./pages/NotFound.tsx";
import SimulatePage from "./pages/SimulatePage";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <CascadeProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route element={<AppShell />}>
              <Route path="/" element={<Navigate to="/simulate" replace />} />
              <Route path="/simulate" element={<SimulatePage />} />
              <Route path="/analytics" element={<AnalyticsPage />} />
              <Route path="/history" element={<HistoryPage />} />
              <Route path="/dataset" element={<DatasetPage />} />
              <Route path="/about" element={<AboutPage />} />
            </Route>
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </CascadeProvider>
  </QueryClientProvider>
);

export default App;
