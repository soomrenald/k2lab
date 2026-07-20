import { useEffect, useState } from "react";
import type {
  CapabilityManifest,
  CredentialStatus,
  GpuOption,
  WorkspaceRecord,
} from "./api";
import { controlPlane } from "./api";
import { CloudOnboarding } from "./components/CloudOnboarding";
import { Icon } from "./components/Icon";
import { WorkspaceStudio } from "./components/WorkspaceStudio";

interface BootstrapState {
  capabilities: CapabilityManifest;
  credential: CredentialStatus;
  gpus: GpuOption[];
  workspace: WorkspaceRecord | null;
}

export function App() {
  const [state, setState] = useState<BootstrapState | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function bootstrap() {
      try {
        const [capabilities, credential, workspaces] = await Promise.all([
          controlPlane.capabilities(),
          controlPlane.credentialStatus(),
          controlPlane.workspaces(),
        ]);
        const gpus = credential.configured ? await controlPlane.gpus() : [];
        if (!cancelled) {
          setState({
            capabilities,
            credential,
            gpus,
            workspace: workspaces.find((item) => item.state !== "deleted") ?? null,
          });
        }
      } catch (caught) {
        if (!cancelled) setError(caught instanceof Error ? caught.message : "Could not reach the control plane");
      }
    }
    bootstrap();
    return () => { cancelled = true; };
  }, []);

  if (error) {
    return (
      <main className="fatal-shell">
        <div className="danger-icon"><Icon name="cloud" /></div>
        <p className="kicker">Control plane unavailable</p>
        <h1>Could not start K2 Region Lab</h1>
        <p>{error}</p>
        <code>uv run k2lab-web --reload</code>
        <button className="primary-button" onClick={() => window.location.reload()}>Try again</button>
      </main>
    );
  }

  if (!state) {
    return (
      <main className="loading-shell">
        <span className="loading-mark">K2</span>
        <span className="loading-bar"><i /></span>
        <p>Opening cloud studio…</p>
      </main>
    );
  }

  if (!state.workspace) {
    return (
      <div className="entry-shell">
        <header className="entry-header">
          <div className="brand-lockup"><span className="brand-mark">K2</span><span><strong>Region Lab</strong><small>Cloud studio</small></span></div>
          <div className="entry-meta"><span>Project schema v{state.capabilities.project_schema_version}</span><span className="divider" /> Persistent-Pod preview</div>
        </header>
        <CloudOnboarding
          capabilities={state.capabilities}
          credential={state.credential}
          gpus={state.gpus}
          onCredential={(credential, gpus) => setState({ ...state, credential, gpus })}
          onWorkspace={(workspace) => setState({ ...state, workspace })}
        />
      </div>
    );
  }

  return (
    <WorkspaceStudio
      workspace={state.workspace}
      developmentBackend={state.capabilities.development_backend}
      onWorkspace={(workspace) => setState({ ...state, workspace })}
      onDelete={() => setState({ ...state, workspace: null })}
    />
  );
}
