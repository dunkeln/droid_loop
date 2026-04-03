<script lang="ts">
	type LiveStatus = {
		running: 'scan' | 'label' | null;
		scan: {
			paused: boolean;
			queue_size: number;
			queue_capacity: number;
			load_eps_per_min: number;
			score_eps_per_min: number;
			current_episode: number | null;
			rss_mb: number;
		};
	};

	let {
		jobRunning,
		clipsSize,
		scanStatus,
		scanning,
		scanPhase,
		currentEpisode,
		liveStatus,
		vlmBusy,
		latestVlmPreview,
		exportMsg,
		onScan,
		onLabel,
		onExport,
		onOpenLatestChat,
		scanButtonLabel
	} = $props<{
		jobRunning: 'scan' | 'label' | null;
		clipsSize: number;
		scanStatus: string;
		scanning: boolean;
		scanPhase: 'idle' | 'loading' | 'scoring';
		currentEpisode: number | null;
		liveStatus: LiveStatus | null;
		vlmBusy: boolean;
		latestVlmPreview: string | null;
		exportMsg: string | null;
		onScan: () => void;
		onLabel: () => void;
		onExport: () => void;
		onOpenLatestChat: () => void;
		scanButtonLabel: () => string;
	}>();

	function isVlmStatus(value: string): boolean {
		return value.startsWith('vlm ');
	}
</script>

<div class="topbar">
	<div class="ctrl-group">
		<button class="ctrl-btn" disabled={jobRunning === 'label'} onclick={onScan}>
			{scanButtonLabel()}
		</button>
		<button class="ctrl-btn" disabled={jobRunning !== null || clipsSize === 0} onclick={onLabel}>
			{#if jobRunning === 'label'}<span class="spinner"></span>labeling…{:else}label{/if}
		</button>
		<button class="ctrl-btn" disabled={jobRunning !== null || clipsSize === 0} onclick={onExport}>
			export
		</button>
	</div>
	{#if scanStatus && !isVlmStatus(scanStatus)}
		<span class="status-line">{scanStatus}</span>
	{/if}
	{#if scanning}
		<span class="status-line phase">{scanPhase} · ep {currentEpisode ?? '—'}</span>
	{/if}
	{#if liveStatus?.running === 'scan' && liveStatus.scan.paused}
		<span class="status-line phase">paused</span>
	{/if}
	{#if liveStatus?.running === 'scan'}
		<span class="status-line live">
			q {liveStatus.scan.queue_size}/{liveStatus.scan.queue_capacity}
			· load {liveStatus.scan.load_eps_per_min}/min
			· score {liveStatus.scan.score_eps_per_min}/min
			· cur ep {liveStatus.scan.current_episode ?? '—'}
			· rss {liveStatus.scan.rss_mb} MB
		</span>
	{/if}
	{#if vlmBusy}
		<span class="status-line phase">vlm querying…</span>
	{/if}
	{#if latestVlmPreview}
		<button class="status-line status-link" onclick={onOpenLatestChat} title="Open latest chat">
			<strong class="status-link-label">incident</strong> {latestVlmPreview}
		</button>
	{/if}
	{#if exportMsg}
		<span class="export-msg" class:ok={exportMsg.startsWith('✓')}>{exportMsg}</span>
	{/if}
</div>

<style>
	.topbar {
		display:flex; align-items:center; gap:12px;
		padding:14px 20px 0; flex-shrink:0;
	}
	.ctrl-group { display:flex; gap:6px; }
	.ctrl-btn {
		display:flex; align-items:center; gap:5px;
		padding:4px 10px; border-radius:10px;
		border:1px solid transparent;
		background:transparent;
		color:rgba(255,255,255,0.42);
		font-size:11px; font-weight:500; letter-spacing:0.01em;
		font-family:inherit; cursor:pointer;
		transition:background 0.12s, color 0.12s, border-color 0.12s, transform 0.12s;
	}
	.ctrl-btn:hover:not(:disabled) {
		background:rgba(255,255,255,0.035);
		border-color:rgba(255,255,255,0.08);
		color:rgba(255,255,255,0.78);
	}
	.ctrl-btn:active:not(:disabled) { transform:translateY(0.5px); }
	.ctrl-btn:focus-visible {
		outline:none;
		border-color:rgba(255,255,255,0.18);
		background:rgba(255,255,255,0.045);
		color:rgba(255,255,255,0.86);
	}
	.ctrl-btn:disabled { opacity:0.26; cursor:not-allowed; }
	.status-line { font-size:11.5px; font-family:monospace; color:rgba(255,255,255,0.45); }
	.status-line.phase { color:rgba(120,200,255,0.78); }
	.status-line.live { color:rgba(255,255,255,0.58); white-space:nowrap; }
	.status-link {
		border:none;
		background:transparent;
		padding:0;
		cursor:pointer;
	}
	.status-link-label {
		font-weight:700;
		color:rgba(255,255,255,0.78);
	}
	.status-link:hover { color:rgba(255,255,255,0.72); }
	.export-msg  { font-size:11.5px; color:rgba(255,255,255,0.3); }
	.export-msg.ok { color:rgba(120,220,140,0.8); }
	.spinner { width:10px; height:10px; border:1.5px solid rgba(255,255,255,0.2); border-top-color:rgba(255,255,255,0.8); border-radius:50%; animation:spin 0.7s linear infinite; }
	@keyframes spin { to { transform:rotate(360deg); } }
</style>
