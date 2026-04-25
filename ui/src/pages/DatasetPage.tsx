import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { apiClient } from "@/api/client";
import { StateBadge } from "@/components/cascade/StateBadge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import type { DatasetRow } from "@/types";

export default function DatasetPage() {
  const [taskFilter, setTaskFilter] = useState("all");
  const [actionFilter, setActionFilter] = useState("all");
  const [selectedRow, setSelectedRow] = useState<DatasetRow | null>(null);

  const statsQuery = useQuery({
    queryKey: ["dataset-stats"],
    queryFn: apiClient.getDatasetStats,
  });
  const rowsQuery = useQuery({
    queryKey: ["dataset-rows"],
    queryFn: apiClient.getDatasetRows,
  });

  const filteredRows = useMemo(() => {
    const rows = rowsQuery.data ?? [];
    return rows.filter((row) => {
      const matchesTask = taskFilter === "all" || row.task === taskFilter;
      const matchesAction = actionFilter === "all" || row.action === actionFilter;
      return matchesTask && matchesAction;
    });
  }, [actionFilter, rowsQuery.data, taskFilter]);

  if (statsQuery.isLoading || rowsQuery.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-24 rounded-[28px]" />
        <Skeleton className="h-[480px] rounded-[28px]" />
      </div>
    );
  }

  if (statsQuery.isError || rowsQuery.isError) {
    return (
      <Alert className="border-[hsl(var(--red))]/30 bg-[hsl(var(--red))]/8">
        <AlertTitle>Could not load dataset browser</AlertTitle>
        <AlertDescription className="flex items-center gap-3">
          <span>The backend may be unavailable or the CSV could not be parsed.</span>
          <Button size="sm" variant="outline" onClick={() => {
            void statsQuery.refetch();
            void rowsQuery.refetch();
          }}>
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  const stats = statsQuery.data;
  const rows = rowsQuery.data;
  const taskOptions = [...new Set(rows.map((row) => row.task))];
  const actionOptions = [...new Set(rows.map((row) => row.action))];

  return (
    <div className="space-y-6">
      <div>
        <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-foreground/45">SFT Dataset Browser</div>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">155 training examples with chain-of-thought</h1>
      </div>

      <Card className="rounded-[28px] border-black/10 bg-white/88 p-5 shadow-[0_24px_60px_-36px_rgba(0,0,0,0.4)]">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap gap-3">
            <StateBadge label={`${stats.total} examples`} tone="info" />
            <StateBadge label={`avg score ${stats.avg_score.toFixed(3)}`} tone="success" />
            <StateBadge label={`${Object.keys(stats.task_distribution).length} tasks`} tone="neutral" />
            <StateBadge label={`${Object.keys(stats.action_distribution).length} action types`} tone="neutral" />
          </div>
          <Button onClick={() => void apiClient.downloadDatasetCsv()}>Download CSV</Button>
        </div>
      </Card>

      <Card className="rounded-[28px] border-black/10 bg-white/88 p-5 shadow-[0_24px_60px_-36px_rgba(0,0,0,0.4)]">
        <div className="mb-4 flex flex-wrap gap-4">
          <div className="w-[220px]">
            <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/45">Filter by task</div>
            <Select value={taskFilter} onValueChange={setTaskFilter}>
              <SelectTrigger>
                <SelectValue placeholder="All tasks" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All tasks</SelectItem>
                {taskOptions.map((task) => (
                  <SelectItem key={task} value={task}>
                    {task}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="w-[220px]">
            <div className="mb-2 font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/45">Filter by action</div>
            <Select value={actionFilter} onValueChange={setActionFilter}>
              <SelectTrigger>
                <SelectValue placeholder="All actions" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All actions</SelectItem>
                {actionOptions.map((action) => (
                  <SelectItem key={action} value={action}>
                    {action}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="overflow-x-auto rounded-2xl border border-black/8">
          <table className="min-w-full text-sm">
            <thead className="bg-black/[0.04] font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/45">
              <tr>
                <th className="px-4 py-3 text-left">ID</th>
                <th className="px-4 py-3 text-left">Task</th>
                <th className="px-4 py-3 text-left">Phase</th>
                <th className="px-4 py-3 text-left">Step</th>
                <th className="px-4 py-3 text-left">Action+Target</th>
                <th className="px-4 py-3 text-left">Score</th>
                <th className="px-4 py-3 text-left">Cascade Depth</th>
              </tr>
            </thead>
            <tbody>
              {filteredRows.map((row) => (
                <tr key={row.id} className="cursor-pointer border-t border-black/8 transition-colors hover:bg-black/[0.03]" onClick={() => setSelectedRow(row)}>
                  <td className="px-4 py-3 font-mono text-[12px]">{row.id}</td>
                  <td className="px-4 py-3">{row.task}</td>
                  <td className="px-4 py-3">{row.phase}</td>
                  <td className="px-4 py-3">{row.step}</td>
                  <td className="px-4 py-3 font-medium">
                    {row.action} {row.target}
                  </td>
                  <td className="px-4 py-3">{Number(row.episode_score).toFixed(3)}</td>
                  <td className="px-4 py-3">{row.cascade_depth}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Dialog open={!!selectedRow} onOpenChange={(open) => !open && setSelectedRow(null)}>
        <DialogContent className="max-w-4xl rounded-[28px]">
          {selectedRow ? (
            <>
              <DialogHeader>
                <DialogTitle>{selectedRow.id} • {selectedRow.task} • {selectedRow.action} {selectedRow.target}</DialogTitle>
              </DialogHeader>
              <div className="grid gap-4 lg:grid-cols-3">
                <Card className="rounded-2xl border-black/10 bg-black/[0.03] p-4">
                  <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/45">Prompt</div>
                  <pre className="mt-3 whitespace-pre-wrap font-mono text-[12px] leading-6 text-foreground/75">{selectedRow.prompt}</pre>
                </Card>
                <Card className="rounded-2xl border-black/10 bg-black/[0.03] p-4">
                  <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/45">Thinking</div>
                  <pre className="mt-3 whitespace-pre-wrap font-mono text-[12px] leading-6 text-foreground/75">{selectedRow.thinking}</pre>
                </Card>
                <Card className="rounded-2xl border-black/10 bg-black/[0.03] p-4">
                  <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-foreground/45">Completion</div>
                  <pre className="mt-3 whitespace-pre-wrap font-mono text-[12px] leading-6 text-foreground/75">{selectedRow.completion}</pre>
                </Card>
              </div>
            </>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}
