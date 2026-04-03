<script lang="ts">
	import { onMount } from 'svelte';
	import { AreaY, Line, Plot } from 'svelteplot';
	import TopBar from '$lib/components/TopBar.svelte';
	import EpisodesSidebar from '$lib/components/EpisodesSidebar.svelte';
	import ChatLog from '$lib/components/ChatLog.svelte';

	const API  = 'http://localhost:8000/api';
	const FURL = (ep: number, fi: number) => `${API}/frames/${ep}_${fi}.jpg`;

	// ── Types ─────────────────────────────────────────────────────────────────
	type Moment = {
		episode_id: number; frame_index: number; cluster_id: number;
		cluster_size?: number; cluster_span?: number[]; context_window?: number[]; label: string;
	};
	type FlaggedFrame = { frame_index: number; cluster_id: number; frame_url: string; };
	type FrameView = { camera: string; frame_url: string; };
	type Clip = {
		episode_id: number;
		preview_urls: string[];       // for camera playback
		flagged: FlaggedFrame[];      // surfaced anomalies
		moments: Moment[];            // loaded from catalog
		descriptions: Record<string, string>;
		video_url?: string;
	};
	type IncidentGroup = { cluster_id: number; moments: Moment[]; representative: Moment };
	type FlagSegment = { start: number; end: number };
	type FlaggedAreaPoint = { frame_index: number; score: number | null };
	type EpisodeScorePoint = {
		frame_index: number;
		score: number;
		cluster_id: number;
		flagged: boolean;
	};
	type SseEvent = {
		type: 'episode_frames'|'frame'|'progress'|'done'|'error'|'ping';
		episode_id?: number; frame_index?: number; cluster_id?: number;
		cluster_size?: number; frame_url?: string; frame_urls?: string[];
		preview_groups?: Array<Record<string, string>>;  // per-camera frames per step
		video_url?: string;
		camera_videos?: Record<string, string>;
		total_frames?: number; frames?: number; flagged?: number;
		total_flagged?: number; total_clusters?: number; message?: string;
	};
	type ScanStatus = {
		active: boolean;
		paused: boolean;
		started_at: number | null;
		queue_size: number;
		queue_capacity: number;
		current_episode: number | null;
		loaded_episodes: number;
		scored_episodes: number;
		loaded_frames: number;
		scored_frames: number;
		total_flagged: number;
		elapsed_s: number;
		load_eps_per_min: number;
		score_eps_per_min: number;
		load_frames_per_s: number;
		score_frames_per_s: number;
		rss_mb: number;
	};
	type ApiStatus = {
		running: 'scan'|'label'|null;
		scan: ScanStatus;
	};
	type VlmQueryResponse = {
		cached: boolean;
		model: string;
		attempts: number;
		result: {
			incident_type: string;
			failure_mode: string;
			confidence: number;
			summary: string;
			evidence_frame_indices: number[];
			actionability: string;
			recommendations: string[];
		};
		raw_text: string;
		token_usage?: {
			input_tokens: number;
			output_tokens: number;
			total_tokens: number;
			accumulated_total_tokens: number;
		};
		context?: {
			history_tokens_est: number;
			history_compacted: boolean;
			history_messages: number;
			context_token_limit: number;
		};
		resolved_clip?: {
			id: string;
			ordinal: number;
			start_frame_index: number;
			end_frame_index: number;
			anchor_frame_index: number;
			anchor_timestamp_s?: number;
			start_timestamp_s?: number;
			end_timestamp_s?: number;
		} | null;
		episode_clips?: {
			id: string;
			ordinal: number;
			start_frame_index: number;
			end_frame_index: number;
		}[];
	};
	type ChatTurn = { role: 'user' | 'assistant'; content: string; ts: number };
	type EpisodeSidebarCard = {
		episode_id: number;
		thumbnail: string | null;
		incidents: number;
		flagged: number;
		scanning: boolean;
	};
	type EpisodeSummary = {
		episode_id: number;
		frame_count?: number;
		flagged_count?: number;
		scanned_at?: number;
	};
	type EpisodeClipRef = {
		id: string;
		ordinal: number;
		start_frame_index: number;
		end_frame_index: number;
		anchor_frame_index: number;
	};

	// ── State ─────────────────────────────────────────────────────────────────
	let clips     = $state<Map<number, Clip>>(new Map());
	let activeEp  = $state<number|null>(null);          // episode being reviewed
	let selMoment = $state<Moment|null>(null);
	let validating= $state<Record<string,boolean>>({});
	let detailViews = $state<FrameView[]>([]);
	let detailViewsLoading = $state(false);
	let detailViewsReq = 0;

	// Camera / playback
	let frameBuf    = $state<string[]>([]);
	let displayFrame= $state('');
	// Per-camera grouped scan frames — each entry is {cam_tag: url} for one timestep
	let scanFrameGroups = $state<Array<Record<string, string>>>([]);
	// Currently displayed scan frame per camera tag
	let scanFrameDisplay = $state<Record<string, string>>({});
	const SCAN_PLAYBACK_FPS = 8;
	let scanning    = $state(false);
	let scanPaused  = $state(false);
	let scanStatus  = $state('');
	let currentEpisode = $state<number|null>(null);
	let scanPhase = $state<'idle'|'loading'|'scoring'>('idle');
	let playTimer: ReturnType<typeof setInterval>|null = null;
	let episodeFrame = $state<number|null>(null);
	let episodeScores = $state<Record<number, EpisodeScorePoint[]>>({});
	let episodeScoresLoading = $state(false);
	let episodeScoresReq = 0;
	let episodeVideoUrls = $state<Record<number, string>>({});
	let episodeCameraVideoUrls = $state<Record<number, Record<string, string>>>({});
	let featuredCameraByEpisode = $state<Record<number, string>>({});
	let cameraVideoEls = $state<Record<string, HTMLVideoElement>>({});
	let episodeVideoReq = 0;
	let episodeCameraVideoReq = 0;
	let episodeVideoEl = $state<HTMLVideoElement | null>(null);
	let mountedVideoEpisode = $state<number | null>(null);
	let episodeDuration = $state(0);
	let episodeCurrentTime = $state(0);
	let videoPlaying = $state(false);
	let videoMuted = $state(false);
	let videoLoop = $state(false);
	let videoRate = $state(1);
	let showMomentInfo = $state(false);
	let videoAnimating = false;
	let videoRaf: number | null = null;
	let timelineScrubbing = $state(false);
	let liveStatus = $state<ApiStatus | null>(null);
	let vlmBusy = $state(false);
	let vlmReq = 0;
	let lastVlmResponse = $state<VlmQueryResponse | null>(null);
	let chatHistory = $state<ChatTurn[]>([]);
	const CHAT_HISTORY_KEY = 'droid:chat-history:v1';
	const CHAT_HISTORY_MAX = 20;
	const CHAT_CONTEXT_TOKEN_LIMIT = 1200;
	let episodeClipRefs = $state<Record<number, EpisodeClipRef[]>>({});
	let episodeClipReq = 0;
	let catalogLoadReq = 0;
	let inspectorOpen = $state(false);
	let inspectorExpanded = $state(false);
	let inspectorQuestion = $state('');
	let inspectorAnswer = $state('');
	let inspectorTimer: ReturnType<typeof setTimeout> | null = null;
	let viewportMode = $state<'main' | 'chat'>('main');

	// Jobs
	let jobRunning = $state<'scan'|'label'|null>(null);
	let exportMsg  = $state<string|null>(null);

	let es: EventSource|null = null;
	let esReconnectTimer: ReturnType<typeof setTimeout> | null = null;

	onMount(() => {
		void loadCatalog();
		try {
			const raw = localStorage.getItem(CHAT_HISTORY_KEY);
			if (raw) {
				const parsed = JSON.parse(raw) as ChatTurn[];
				if (Array.isArray(parsed)) chatHistory = parsed.slice(-CHAT_HISTORY_MAX);
			}
		} catch {
			chatHistory = [];
		}
		const onChatSubmit = (event: Event) => {
			const detail = (event as CustomEvent<{ message?: string }>).detail;
			const message = detail?.message?.trim();
			if (!message) return;
			void runVlmQuery(message);
		};
		const onGlobalKeydown = (event: KeyboardEvent) => {
			const target = event.target as HTMLElement | null;
			const isTypingTarget =
				target?.tagName === 'INPUT' ||
				target?.tagName === 'TEXTAREA' ||
				(target?.isContentEditable ?? false);
				if (isTypingTarget) return;
				if (!activeClip) return;
				if (!resolveEpisodeVideoTarget()) return;
				if (event.key === 'ArrowLeft') {
					event.preventDefault();
					seekVideo(-1);
				} else if (event.key === 'ArrowRight') {
					event.preventDefault();
					seekVideo(1);
				}
			};
		window.addEventListener('droid:chat-submit', onChatSubmit as EventListener);
		window.addEventListener('keydown', onGlobalKeydown);
		const onViewMode = (event: Event) => {
			const detail = (event as CustomEvent<{ view?: 'main' | 'chat' }>).detail;
			if (detail?.view === 'main' || detail?.view === 'chat') viewportMode = detail.view;
		};
		window.addEventListener('droid:view-mode', onViewMode as EventListener);
		void refreshStatus(true);
		return () => {
			window.removeEventListener('droid:chat-submit', onChatSubmit as EventListener);
			window.removeEventListener('keydown', onGlobalKeydown);
			window.removeEventListener('droid:view-mode', onViewMode as EventListener);
			es?.close();
			if (esReconnectTimer) clearTimeout(esReconnectTimer);
			stopVideoAnimationLoop();
			if (inspectorTimer) clearTimeout(inspectorTimer);
		};
	});

	function pushChatTurn(turn: ChatTurn) {
		chatHistory = [...chatHistory, turn].slice(-CHAT_HISTORY_MAX);
		try {
			localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(chatHistory));
		} catch {
			// ignore storage failures
		}
	}

	function setInspectorActive() {
		inspectorOpen = true;
		if (inspectorTimer) clearTimeout(inspectorTimer);
		inspectorTimer = setTimeout(() => {
			inspectorExpanded = false;
		}, 8000);
	}

	function toggleInspectorExpanded() {
		inspectorExpanded = !inspectorExpanded;
		setInspectorActive();
	}

	$effect(() => {
		const m = selMoment;
		showMomentInfo = false;
		if (!m) {
			detailViews = [];
			detailViewsLoading = false;
			return;
		}
		const req = ++detailViewsReq;
		detailViewsLoading = true;
		void (async () => {
			try {
				const r = await fetch(`${API}/frame-views/${m.episode_id}/${m.frame_index}`);
				const views: FrameView[] = r.ok ? await r.json() : [];
				const mapped = views.map((v) => ({
					camera: v.camera,
					frame_url: v.frame_url.startsWith('http') ? v.frame_url : `http://localhost:8000${v.frame_url}`
				}));
				const fallback = [{ camera: 'primary', frame_url: FURL(m.episode_id, m.frame_index) }];
				if (req === detailViewsReq) detailViews = mapped.length > 0 ? mapped : fallback;
			} catch {
				if (req === detailViewsReq) {
					detailViews = [{ camera: 'primary', frame_url: FURL(m.episode_id, m.frame_index) }];
				}
			} finally {
				if (req === detailViewsReq) detailViewsLoading = false;
			}
		})();
	});

	// ── FPS playback ──────────────────────────────────────────────────────────
	function startPlayback() {
		stopPlayback();
		console.log('[droid-loop] startPlayback at', SCAN_PLAYBACK_FPS, 'fps');
		playTimer = setInterval(() => {
			// Prefer grouped (all-camera) frames over legacy single-camera buffer
			if (scanFrameGroups.length > 0) {
				const group = scanFrameGroups[0];
				scanFrameGroups = scanFrameGroups.slice(1);
				scanFrameDisplay = { ...scanFrameDisplay, ...group };
				// Keep displayFrame in sync for any code still referencing it
				const primaryUrl = Object.values(group)[0];
				if (primaryUrl) displayFrame = `http://localhost:8000${primaryUrl.startsWith('/') ? primaryUrl : '/' + primaryUrl}`;
			} else if (frameBuf.length > 0) {
				displayFrame = frameBuf[0];
				frameBuf = frameBuf.slice(1);
			}
		}, 1000 / SCAN_PLAYBACK_FPS);
	}
	function stopPlayback() {
		if (playTimer) { clearInterval(playTimer); playTimer = null; }
	}

	$effect(() => {
		activeEp;
		activeClip;
		scanning;
		selMoment;
		if (scanning || !activeClip) {
			episodeFrame = null;
			return;
		}
		if (selMoment) {
			episodeFrame = selMoment.frame_index;
			return;
		}
		const seq = episodePlaybackFrames(activeClip);
		episodeFrame = seq[0] ?? activeClip.moments[0]?.frame_index ?? null;
	});

	$effect(() => {
		const clip = activeClip;
		scanning;
		if (!clip) return;
		if (episodeScores[clip.episode_id]?.length) return;
		const req = ++episodeScoresReq;
		episodeScoresLoading = true;
		void (async () => {
			try {
				const r = await fetch(`${API}/episode-scores/${clip.episode_id}`);
				const rows: EpisodeScorePoint[] = r.ok ? await r.json() : [];
				if (req === episodeScoresReq) {
					episodeScores = {
						...episodeScores,
						[clip.episode_id]: rows.sort((a, b) => a.frame_index - b.frame_index)
					};
				}
			} catch {
				if (req === episodeScoresReq) {
					episodeScores = { ...episodeScores, [clip.episode_id]: [] };
				}
			} finally {
				if (req === episodeScoresReq) episodeScoresLoading = false;
			}
		})();
	});

	$effect(() => {
		const clip = activeClip;
		if (!clip) return;
		if (episodeCameraVideoUrls[clip.episode_id]) return;
		const req = ++episodeCameraVideoReq;
		void (async () => {
			try {
				const r = await fetch(`${API}/episode-videos/${clip.episode_id}`);
				if (!r.ok) return;
				const payload = (await r.json()) as Record<string, string>;
				if (req !== episodeCameraVideoReq) return;
				const full: Record<string, string> = {};
				for (const [camera, raw] of Object.entries(payload ?? {})) {
					full[camera] = raw.startsWith('http') ? raw : `http://localhost:8000${raw}`;
				}
				if (Object.keys(full).length === 0) return;
				episodeCameraVideoUrls = { ...episodeCameraVideoUrls, [clip.episode_id]: full };
			} catch {
				// ignore missing multi-camera videos
			}
		})();
	});

	$effect(() => {
		const clip = activeClip;
		if (!clip) return;
		if (episodeVideoUrls[clip.episode_id]) return;
		const req = ++episodeVideoReq;
		void (async () => {
			try {
				const r = await fetch(`${API}/episode-video/${clip.episode_id}`);
				if (!r.ok) return;
				const payload = await r.json();
				const raw = payload?.video_url as string | undefined;
				if (!raw || req !== episodeVideoReq) return;
				const full = raw.startsWith('http') ? raw : `http://localhost:8000${raw}`;
				episodeVideoUrls = { ...episodeVideoUrls, [clip.episode_id]: full };
			} catch {
				// ignore missing server video
			}
		})();
	});

	$effect(() => {
		const clip = activeClip;
		if (!clip) return;
		if (episodeClipRefs[clip.episode_id]) return;
		const req = ++episodeClipReq;
		void (async () => {
			try {
				const r = await fetch(`${API}/episode-clips/${clip.episode_id}`);
				const rows: EpisodeClipRef[] = r.ok ? await r.json() : [];
				if (req !== episodeClipReq) return;
				episodeClipRefs = { ...episodeClipRefs, [clip.episode_id]: rows };
			} catch {
				if (req !== episodeClipReq) return;
				episodeClipRefs = { ...episodeClipRefs, [clip.episode_id]: [] };
			}
		})();
	});

	$effect(() => {
		const clip = activeClip;
		const ep = clip?.episode_id ?? null;
		if (ep === mountedVideoEpisode) return;
		mountedVideoEpisode = ep;
		cameraVideoEls = {};
		episodeVideoEl = null;
		episodeDuration = 0;
		episodeCurrentTime = 0;
		stopVideoAnimationLoop();
	});

	$effect(() => {
		const clip = activeClip;
		if (!clip) return;
		const cams = episodeCameraEntries(clip);
		if (cams.length === 0) return;
		const cur = featuredCameraByEpisode[clip.episode_id];
		if (cur && cams.some(([camera]) => camera === cur)) return;
		featuredCameraByEpisode = { ...featuredCameraByEpisode, [clip.episode_id]: cams[0][0] };
	});

	$effect(() => {
		const clip = activeClip;
		if (!clip) return;
		const camera = featuredCameraByEpisode[clip.episode_id];
		if (!camera) return;
		const el = cameraVideoEls[camera];
		if (!el) return;
		episodeVideoEl = el;
		episodeDuration = el.duration || episodeDuration;
		episodeCurrentTime = el.currentTime || episodeCurrentTime;
		syncVideoUiState();
	});

	// ── Load existing catalog into clips ──────────────────────────────────────
	async function loadCatalog() {
		const req = ++catalogLoadReq;
		const [episodesRes, momRes, descRes] = await Promise.all([
			fetch(`${API}/episodes`), fetch(`${API}/catalog`), fetch(`${API}/descriptions`)
		]);
		if (req !== catalogLoadReq) return;
		const episodes: EpisodeSummary[] = episodesRes.ok ? await episodesRes.json() : [];
		const moments: Moment[] = await momRes.json();
		const allDescs: Record<string, Record<string,string>> = await descRes.json();

		const map = new Map<number, Clip>();
		for (const ep of episodes) {
			map.set(ep.episode_id, {
				episode_id: ep.episode_id,
				preview_urls: [],
				flagged: [],
				moments: [],
				descriptions: allDescs[String(ep.episode_id)] ?? {},
			});
		}
		for (const m of moments) {
			if (!map.has(m.episode_id)) {
				map.set(m.episode_id, {
					episode_id: m.episode_id,
					preview_urls: [],
					flagged: [],
					moments: [],
					descriptions: allDescs[String(m.episode_id)] ?? {},
				});
			}
			const clip = map.get(m.episode_id)!;
			clip.moments.push(m);
			if (!clip.flagged.find(f => f.frame_index === m.frame_index)) {
				clip.flagged.push({
					frame_index: m.frame_index,
					cluster_id: m.cluster_id,
					frame_url: FURL(m.episode_id, m.frame_index),
				});
			}
		}
		if (req !== catalogLoadReq) return;
		clips = map;
	}

	function closeEventStream() {
		es?.close();
		es = null;
		if (esReconnectTimer) {
			clearTimeout(esReconnectTimer);
			esReconnectTimer = null;
		}
	}

	async function refreshStatus(reconnectScan = false) {
		try {
			const r = await fetch(`${API}/status`);
			if (!r.ok) return;
			const status = (await r.json()) as ApiStatus;
			liveStatus = status;
			jobRunning = status.running;
			scanning = status.running === 'scan';
			scanPaused = status.scan.paused;
			currentEpisode = status.scan.current_episode;
			if (status.running === 'scan') {
				if (status.scan.current_episode !== null) {
					void hydrateEpisodeMedia(status.scan.current_episode);
				}
				scanPhase = status.scan.paused ? 'idle' : 'scoring';
				scanStatus = status.scan.paused
					? 'paused'
					: status.scan.current_episode !== null
						? `ep ${status.scan.current_episode} · scanning`
						: 'scan running';
				if (!status.scan.paused) startPlayback();
				else stopPlayback();
				if (reconnectScan && !es) {
					openEs(`${API}/scan/events`, handleScanEvent, 'scan');
				}
				return;
			}
			if (status.running === 'label') {
				scanStatus = 'querying VLM…';
				return;
			}
			stopPlayback();
		} catch {
			// ignore transient status failures
		}
	}

	async function hydrateEpisodeMedia(episodeId: number) {
		if (!episodeVideoUrls[episodeId]) {
			try {
				const r = await fetch(`${API}/episode-video/${episodeId}`);
				if (r.ok) {
					const payload = await r.json();
					const raw = payload?.video_url as string | undefined;
					if (raw) {
						episodeVideoUrls = {
							...episodeVideoUrls,
							[episodeId]: raw.startsWith('http') ? raw : `http://localhost:8000${raw}`
						};
					}
				}
			} catch {
				// ignore media hydration failures
			}
		}
		if (!episodeCameraVideoUrls[episodeId]) {
			try {
				const r = await fetch(`${API}/episode-videos/${episodeId}`);
				if (!r.ok) return;
				const payload = (await r.json()) as Record<string, string>;
				const mapped: Record<string, string> = {};
				for (const [camera, raw] of Object.entries(payload ?? {})) {
					mapped[camera] = raw.startsWith('http') ? raw : `http://localhost:8000${raw}`;
				}
				if (Object.keys(mapped).length > 0) {
					episodeCameraVideoUrls = { ...episodeCameraVideoUrls, [episodeId]: mapped };
				}
			} catch {
				// ignore media hydration failures
			}
		}
	}

	// ── SSE helpers ───────────────────────────────────────────────────────────
	function openEs(url: string, onEvt: (e: SseEvent) => void, kind: 'scan' | 'label') {
		closeEventStream();
		es = new EventSource(url);
		const evtQueue: SseEvent[] = [];
		let pumping = false;
		const pump = async () => {
			if (pumping) return;
			pumping = true;
				while (evtQueue.length > 0) {
					const e = evtQueue.shift()!;
					onEvt(e);
					if (e.type === 'done' || e.type === 'error') { closeEventStream(); jobRunning = null; }
				if (evtQueue.length > 0) {
					await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
				}
			}
			pumping = false;
		};
		es.onmessage = (raw) => {
			const e: SseEvent = JSON.parse(raw.data);
			evtQueue.push(e);
			void pump();
		};
		es.onerror = () => {
			closeEventStream();
			if (kind !== 'scan') {
				jobRunning = null;
				return;
			}
			esReconnectTimer = setTimeout(() => {
				void refreshStatus(true);
			}, 400);
		};
	}

	// ── Scan ──────────────────────────────────────────────────────────────────
	function handleScanEvent(e: SseEvent) {
		if (e.type === 'episode_frames') {
			const urls = (e.frame_urls ?? []).map(u => `http://localhost:8000${u}`);
			if (e.preview_groups && e.preview_groups.length > 0) {
				scanFrameGroups = [...scanFrameGroups, ...e.preview_groups];
			} else {
				frameBuf = [...frameBuf, ...urls];
			}
			scanStatus = `ep ${e.episode_id}  ·  ${e.total_frames} frames  ·  scoring…`;
			currentEpisode = e.episode_id ?? null;
			scanPhase = 'scoring';
			if (!clips.has(e.episode_id!)) {
				clips.set(e.episode_id!, {
					episode_id: e.episode_id!, preview_urls: urls,
					flagged: [], moments: [], descriptions: {},
					video_url: e.video_url ? `http://localhost:8000${e.video_url}` : undefined,
				});
				clips = new Map(clips);
			}
			if (e.video_url && e.episode_id !== undefined) {
				episodeVideoUrls = {
					...episodeVideoUrls,
					[e.episode_id]: `http://localhost:8000${e.video_url}`
				};
			}
			if (e.camera_videos && e.episode_id !== undefined) {
				const mapped: Record<string, string> = {};
				for (const [camera, raw] of Object.entries(e.camera_videos)) {
					mapped[camera] = raw.startsWith('http') ? raw : `http://localhost:8000${raw}`;
				}
				episodeCameraVideoUrls = {
					...episodeCameraVideoUrls,
					[e.episode_id]: mapped
				};
			}
		} else if (e.type === 'frame') {
			const url = `http://localhost:8000${e.frame_url}`;
			const clip = clips.get(e.episode_id!);
			if (clip) {
				clip.flagged = [...clip.flagged, {
					frame_index: e.frame_index!, cluster_id: e.cluster_id!, frame_url: url,
				}];
				clips = new Map(clips);
			}
		} else if (e.type === 'progress') {
			scanStatus = `ep ${e.episode_id}  ·  ${e.flagged}/${e.frames} flagged  ·  total ${e.total_flagged}`;
			currentEpisode = e.episode_id ?? currentEpisode;
			scanPhase = 'scoring';
		} else if (e.type === 'done') {
			scanStatus = `✓ done — ${e.total_flagged} moments cataloged`;
			scanPhase = 'idle'; currentEpisode = null;
			scanning = false; scanPaused = false; stopPlayback();
			void loadCatalog();
			void refreshStatus(false);
		} else if (e.type === 'error') {
			scanPhase = 'idle'; currentEpisode = null;
			scanStatus = `✕ ${e.message}`; scanning = false; scanPaused = false; stopPlayback();
			void refreshStatus(false);
		}
	}

	async function startScan() {
		clips = new Map(); frameBuf = []; displayFrame = ''; scanFrameGroups = []; scanFrameDisplay = {}; exportMsg = null;
		episodeScores = {};
		episodeVideoUrls = {};
		episodeCameraVideoUrls = {};
		featuredCameraByEpisode = {};
		cameraVideoEls = {};
		const r = await fetch(`${API}/scan`, { method:'POST', headers:{'Content-Type':'application/json'}, body:'{}' });
		if (!r.ok) { scanStatus = (await r.json()).detail ?? 'scan failed'; return; }
		jobRunning = 'scan'; scanning = true; scanPaused = false; scanStatus = 'initialising…';
		scanPhase = 'loading'; currentEpisode = null;
		startPlayback();

		openEs(`${API}/scan/events`, handleScanEvent, 'scan');
	}

	async function pauseScan() {
		const r = await fetch(`${API}/scan/pause`, { method: 'POST' });
		if (!r.ok) {
			scanStatus = (await r.json()).detail ?? 'pause failed';
			return;
		}
		scanPaused = true;
		scanStatus = 'paused';
		scanPhase = 'idle';
	}

	async function resumeScan() {
		const r = await fetch(`${API}/scan/resume`, { method: 'POST' });
		if (!r.ok) {
			scanStatus = (await r.json()).detail ?? 'resume failed';
			return;
		}
		scanPaused = false;
		// Resume should deterministically restore the live scan surface.
		viewportMode = 'main';
		selMoment = null;
		activeEp = null;
		showMomentInfo = false;
		detailViews = [];
		detailViewsLoading = false;
		await refreshStatus(true);
	}

	function isScanActive(): boolean {
		return scanning || jobRunning === 'scan';
	}

	function scanButtonLabel(): string {
		if (!isScanActive()) return 'scan';
		return scanPaused ? 'resume' : 'pause';
	}

	async function onScanButton() {
		if (!isScanActive()) {
			await startScan();
			return;
		}
		if (scanPaused) {
			await resumeScan();
			return;
		}
		await pauseScan();
	}

	// ── Label ─────────────────────────────────────────────────────────────────
	async function startLabel() {
		exportMsg = null;
		const r = await fetch(`${API}/label`, { method:'POST', headers:{'Content-Type':'application/json'}, body:'{}' });
		if (!r.ok) { scanStatus = (await r.json()).detail ?? 'label failed'; return; }
		jobRunning = 'label'; scanStatus = 'querying VLM…';
			openEs(`${API}/label/events`, (e) => {
				if (e.type === 'done') {
					scanStatus = `✓ ${e.total_clusters} clusters labeled`;
					void loadCatalog();
				} else if (e.type === 'error') {
					scanStatus = `✕ ${e.message}`;
				}
			}, 'label');
	}

	// ── Export ────────────────────────────────────────────────────────────────
	async function doExport() {
		const r = await fetch(`${API}/export`, { method:'POST' });
		if (!r.ok) { exportMsg = `✕ ${(await r.json()).detail}`; return; }
		const d = await r.json(); exportMsg = `✓ ${d.exported} → ${d.path}`;
	}

	// ── Validate ──────────────────────────────────────────────────────────────
	async function validate(m: Moment, verdict: 'approved'|'rejected') {
		const key = `${m.episode_id}-${m.frame_index}`;
		validating[key] = true;
		await fetch(`${API}/validate`, {
			method:'POST', headers:{'Content-Type':'application/json'},
			body: JSON.stringify({...m, verdict})
		});
		validating[key] = false;
		if (selMoment?.frame_index === m.frame_index) selMoment = null;
		const clip = clips.get(m.episode_id);
		if (clip) {
			clip.moments = clip.moments.filter(x => x.frame_index !== m.frame_index);
			clips = new Map(clips);
		}
	}

	// ── Helpers ───────────────────────────────────────────────────────────────
	function clusterColor(id: number) {
		if (id === -1) return 'rgba(255,80,80,0.15)';
		const p = ['rgba(100,180,255,0.13)','rgba(120,255,160,0.13)','rgba(255,200,80,0.13)','rgba(200,120,255,0.13)','rgba(255,140,100,0.13)'];
		return p[id % p.length];
	}

	function cameraLabel(camera: string) {
		if (camera === 'primary') return 'primary';
		return camera.replaceAll('_', ' ');
	}

	function episodeRepresentativeUrl(clip: Clip): string | null {
		if (clip.flagged.length > 0) return clip.flagged[0].frame_url;
		if (clip.moments.length > 0) return FURL(clip.moments[0].episode_id, clip.moments[0].frame_index);
		return clip.preview_urls[0] ?? null;
	}

	function incidentGroups(clip: Clip): IncidentGroup[] {
		const byCluster = new Map<number, Moment[]>();
		for (const m of clip.moments) {
			if (!byCluster.has(m.cluster_id)) byCluster.set(m.cluster_id, []);
			byCluster.get(m.cluster_id)!.push(m);
		}
		const groups: IncidentGroup[] = [];
		for (const [cluster_id, moments] of byCluster.entries()) {
			moments.sort((a, b) => a.frame_index - b.frame_index);
			groups.push({ cluster_id, moments, representative: moments[0] });
		}
		groups.sort((a, b) => b.moments.length - a.moments.length);
		return groups;
	}

	function episodeBounds(clip: Clip) {
		let min = Number.POSITIVE_INFINITY;
		let max = 0;
		for (const m of clip.moments) {
			min = Math.min(min, m.frame_index);
			max = Math.max(max, m.frame_index);
			if (m.cluster_span?.length === 2) {
				min = Math.min(min, m.cluster_span[0]);
				max = Math.max(max, m.cluster_span[1]);
			}
			for (const fi of m.context_window ?? []) {
				min = Math.min(min, fi);
				max = Math.max(max, fi);
			}
		}
		if (!Number.isFinite(min)) min = 0;
		if (max <= min) max = min + 1;
		return { min, max, span: max - min };
	}

	function episodePlaybackFrames(clip: Clip): number[] {
		const pool = new Set<number>();
		for (const m of clip.moments) {
			pool.add(m.frame_index);
			for (const fi of m.context_window ?? []) pool.add(fi);
			if (m.cluster_span?.length === 2) {
				const lo = Math.max(m.cluster_span[0], m.frame_index - 3);
				const hi = Math.min(m.cluster_span[1], m.frame_index + 3);
				for (let i = lo; i <= hi; i += 1) pool.add(i);
			}
		}
		return [...pool].sort((a, b) => a - b);
	}

	function scoreSeries(clip: Clip): EpisodeScorePoint[] {
		return episodeScores[clip.episode_id] ?? [];
	}

	function smoothedScoreSeries(clip: Clip): EpisodeScorePoint[] {
		const src = scoreSeries(clip);
		if (src.length === 0) return [];
		const out: EpisodeScorePoint[] = [];
		let ema = src[0].score;
		const alpha = 0.28;
		const k = 12; // log compression strength
		for (let i = 0; i < src.length; i += 1) {
			const p = src[i];
			const logNorm = Math.log1p(k * p.score) / Math.log1p(k);
			ema = i === 0 ? logNorm : alpha * logNorm + (1 - alpha) * ema;
			// Peak-preserving blend: smooth while keeping sharp anomalies visible.
			const blended = Math.max(ema, logNorm * 0.9);
			out.push({ ...p, score: blended });
		}
		return out;
	}

	function interpolatedScorePoint(clip: Clip): EpisodeScorePoint | null {
		const series = smoothedScoreSeries(clip);
		if (series.length === 0) return null;
		const frame = currentEpisodeFrame(clip);
		if (frame <= series[0].frame_index) return { ...series[0] };
		if (frame >= series[series.length - 1].frame_index) return { ...series[series.length - 1] };
		for (let i = 1; i < series.length; i += 1) {
			const lo = series[i - 1];
			const hi = series[i];
			if (frame > hi.frame_index) continue;
			const span = Math.max(1, hi.frame_index - lo.frame_index);
			const t = (frame - lo.frame_index) / span;
			return {
				frame_index: frame,
				score: lo.score + (hi.score - lo.score) * t,
				cluster_id: t < 0.5 ? lo.cluster_id : hi.cluster_id,
				flagged: lo.flagged || hi.flagged
			};
		}
		return { ...series[series.length - 1] };
	}

	function playedSmoothedSeries(clip: Clip): EpisodeScorePoint[] {
		const series = smoothedScoreSeries(clip);
		if (series.length === 0) return [];
		const frame = currentEpisodeFrame(clip);
		const played = series.filter((p) => p.frame_index <= frame);
		const head = interpolatedScorePoint(clip);
		if (head && (played.length === 0 || played[played.length - 1].frame_index !== head.frame_index)) {
			played.push(head);
		}
		return played;
	}

	function flaggedSmoothedSeries(clip: Clip): EpisodeScorePoint[] {
		return smoothedScoreSeries(clip).filter((p) => p.flagged);
	}

	function flaggedSegments(clip: Clip): FlagSegment[] {
		const flagged = scoreSeries(clip)
			.filter((p) => p.flagged)
			.sort((a, b) => a.frame_index - b.frame_index);
		if (flagged.length === 0) return [];
		const segments: FlagSegment[] = [];
		let start = flagged[0].frame_index;
		let end = flagged[0].frame_index;
		for (let i = 1; i < flagged.length; i += 1) {
			const fi = flagged[i].frame_index;
			if (fi - end <= 2) {
				end = fi;
				continue;
			}
			segments.push({ start, end });
			start = fi;
			end = fi;
		}
		segments.push({ start, end });
		return segments;
	}

	function flaggedAreaSeries(clip: Clip): FlaggedAreaPoint[] {
		const series = smoothedScoreSeries(clip);
		const segments = flaggedSegments(clip);
		if (series.length === 0 || segments.length === 0) return [];
		return series.map((p) => {
			const inFlaggedWindow = segments.some((seg) => p.frame_index >= seg.start && p.frame_index <= seg.end);
			return { frame_index: p.frame_index, score: inFlaggedWindow ? p.score : null };
		}) as FlaggedAreaPoint[];
	}

	function playheadPercent(clip: Clip): number {
		const { min, span } = episodeBounds(clip);
		const frame = currentEpisodeFrame(clip);
		return Math.max(0, Math.min(100, ((frame - min) / Math.max(1, span)) * 100));
	}

	function seekToPercent(clip: Clip, pct: number) {
		const p = Math.max(0, Math.min(1, pct));
		const lead = resolveEpisodeVideoTarget();
		if (lead && episodeDuration > 0) {
			const next = p * episodeDuration;
			lead.currentTime = next;
			episodeCurrentTime = next;
			return;
		}
		const { min, span } = episodeBounds(clip);
		episodeFrame = Math.round(min + p * span);
	}

	function onTimelinePointerDown(clip: Clip, event: PointerEvent) {
		const el = event.currentTarget as HTMLDivElement;
		const rect = el.getBoundingClientRect();
		const pct = (event.clientX - rect.left) / Math.max(1, rect.width);
		timelineScrubbing = true;
		el.setPointerCapture(event.pointerId);
		seekToPercent(clip, pct);
	}

	function onTimelinePointerMove(clip: Clip, event: PointerEvent) {
		if (!timelineScrubbing) return;
		const el = event.currentTarget as HTMLDivElement;
		const rect = el.getBoundingClientRect();
		const pct = (event.clientX - rect.left) / Math.max(1, rect.width);
		seekToPercent(clip, pct);
	}

	function onTimelinePointerUp(event: PointerEvent) {
		const el = event.currentTarget as HTMLDivElement;
		timelineScrubbing = false;
		if (el.hasPointerCapture(event.pointerId)) el.releasePointerCapture(event.pointerId);
	}

	function stopVideoAnimationLoop() {
		videoAnimating = false;
		if (videoRaf !== null) {
			cancelAnimationFrame(videoRaf);
			videoRaf = null;
		}
	}

	function startVideoAnimationLoop() {
		if (videoAnimating || !episodeVideoEl) return;
		videoAnimating = true;
		const tick = () => {
			if (!videoAnimating || !episodeVideoEl) return;
			if (!episodeVideoEl.paused) {
				episodeCurrentTime = episodeVideoEl.currentTime;
				videoRaf = requestAnimationFrame(tick);
				return;
			}
			stopVideoAnimationLoop();
		};
		videoRaf = requestAnimationFrame(tick);
	}

	function syncVideoUiState() {
		if (!episodeVideoEl) return;
		videoPlaying = !episodeVideoEl.paused;
		videoMuted = episodeVideoEl.muted;
		videoLoop = episodeVideoEl.loop;
		videoRate = episodeVideoEl.playbackRate;
	}

	function resolveEpisodeVideoTarget(): HTMLVideoElement | null {
		if (episodeVideoEl) return episodeVideoEl;
		const clip = activeClip;
		if (clip) {
			const camera = featuredCamera(clip);
			if (camera && cameraVideoEls[camera]) {
				episodeVideoEl = cameraVideoEls[camera];
				syncVideoUiState();
				return episodeVideoEl;
			}
		}
		const fallback = Object.values(cameraVideoEls)[0] ?? null;
		if (fallback) {
			episodeVideoEl = fallback;
			syncVideoUiState();
		}
		return fallback;
	}

	async function toggleVideoPlay() {
		const els = Object.values(cameraVideoEls);
		const lead = resolveEpisodeVideoTarget();
		if (els.length === 0 && !lead) return;
		if (!lead) return;
		if (lead.paused) {
			await Promise.all(els.map(async (el) => {
				try {
					await el.play();
				} catch {
					// ignore autoplay/playback interruptions
				}
			}));
		} else {
			for (const el of els) el.pause();
		}
		syncVideoUiState();
	}

	function seekVideo(deltaSec: number) {
		const els = Object.values(cameraVideoEls);
		const lead = resolveEpisodeVideoTarget();
		if (els.length === 0 && !lead) return;
		if (!lead) return;
		const next = Math.max(0, Math.min(lead.duration || 0, lead.currentTime + deltaSec));
		for (const el of els) {
			const duration = Number.isFinite(el.duration) ? el.duration : lead.duration;
			el.currentTime = Math.max(0, Math.min(duration || 0, next));
		}
		episodeCurrentTime = next;
	}

	function adaptiveSeekStepSec(): number {
		const duration = episodeDuration > 0 ? episodeDuration : (episodeVideoEl?.duration ?? 0);
		if (!Number.isFinite(duration) || duration <= 0) return 10;
		return Math.max(1, Math.min(10, duration * 0.1));
	}

	function toggleVideoLoop() {
		const els = Object.values(cameraVideoEls);
		const lead = resolveEpisodeVideoTarget();
		if (els.length === 0 && !lead) return;
		if (!lead) return;
		const next = !lead.loop;
		for (const el of els) el.loop = next;
		syncVideoUiState();
	}

	function toggleVideoMute() {
		const els = Object.values(cameraVideoEls);
		const lead = resolveEpisodeVideoTarget();
		if (els.length === 0 && !lead) return;
		if (!lead) return;
		const next = !lead.muted;
		for (const el of els) el.muted = next;
		syncVideoUiState();
	}

	function cycleVideoRate() {
		const els = Object.values(cameraVideoEls);
		const lead = resolveEpisodeVideoTarget();
		if (els.length === 0 && !lead) return;
		if (!lead) return;
		const rates = [0.5, 1, 1.25, 1.5, 2];
		const idx = rates.indexOf(lead.playbackRate);
		const next = rates[(idx + 1) % rates.length];
		for (const el of els) el.playbackRate = next;
		syncVideoUiState();
	}

	function episodeCameraEntries(clip: Clip): Array<[string, string]> {
		const fromCameraMap = Object.entries(episodeCameraVideoUrls[clip.episode_id] ?? {});
		if (fromCameraMap.length > 0) return fromCameraMap;
		const primary = episodeVideoUrls[clip.episode_id];
		return primary ? [['primary', primary]] : [];
	}

	function scanOverlayCameraEntries(): Array<[string, string]> {
		if (currentEpisode === null) return [];
		const byCamera = Object.entries(episodeCameraVideoUrls[currentEpisode] ?? {});
		if (byCamera.length === 0) return [];
		const primaryUrl = episodeVideoUrls[currentEpisode];
		return byCamera.filter(([, url]) => !primaryUrl || url !== primaryUrl);
	}

	function featuredCamera(clip: Clip): string | null {
		const selected = featuredCameraByEpisode[clip.episode_id];
		if (!selected) {
			const first = episodeCameraEntries(clip)[0]?.[0];
			return first ?? null;
		}
		if (episodeCameraEntries(clip).some(([camera]) => camera === selected)) return selected;
		return episodeCameraEntries(clip)[0]?.[0] ?? null;
	}

	function selectFeaturedCamera(clip: Clip, camera: string) {
		featuredCameraByEpisode = { ...featuredCameraByEpisode, [clip.episode_id]: camera };
		const el = cameraVideoEls[camera];
		if (el) {
			episodeVideoEl = el;
			episodeDuration = el.duration || episodeDuration;
			episodeCurrentTime = el.currentTime || episodeCurrentTime;
			syncVideoUiState();
		}
	}

	function registerCameraVideo(node: HTMLVideoElement, camera: string) {
		cameraVideoEls = { ...cameraVideoEls, [camera]: node };
		const clip = activeClip;
		if (clip && featuredCamera(clip) === camera) {
			episodeVideoEl = node;
			episodeDuration = node.duration || episodeDuration;
			episodeCurrentTime = node.currentTime || episodeCurrentTime;
			syncVideoUiState();
		}
		return {
			destroy() {
				const next = { ...cameraVideoEls };
				delete next[camera];
				cameraVideoEls = next;
				if (episodeVideoEl === node) episodeVideoEl = null;
			}
		};
	}

	function primaryScanFrameUrl(): string {
		const primary = Object.values(scanFrameDisplay)[0] ?? displayFrame;
		if (!primary) return '';
		return primary.startsWith('http') ? primary : `http://localhost:8000${primary}`;
	}

	function primaryScanVideoUrl(): string {
		if (currentEpisode === null) return '';
		return episodeVideoUrls[currentEpisode] ?? '';
	}

	function scanOverlayVideoEntries(): Array<[string, string]> {
		if (currentEpisode === null) return [];
		const entries = Object.entries(episodeCameraVideoUrls[currentEpisode] ?? {});
		if (entries.length === 0) return [];
		const primary = primaryScanVideoUrl();
		return entries.filter(([, url]) => !primary || url !== primary);
	}

	function currentEpisodeFrame(clip: Clip): number {
		const videoUrl = episodeVideoUrls[clip.episode_id] || episodeCameraEntries(clip).length > 0;
		if (videoUrl && episodeDuration > 0) {
			const { min, span } = episodeBounds(clip);
			return Math.round(min + (episodeCurrentTime / episodeDuration) * span);
		}
		return episodeFrame ?? selMoment?.frame_index ?? clip.moments[0]?.frame_index ?? 0;
	}

	function currentEpisodeClipRef(clip: Clip): EpisodeClipRef | null {
		const refs = episodeClipRefs[clip.episode_id] ?? [];
		if (refs.length === 0) return null;
		const frame = currentEpisodeFrame(clip);
		const inRange = refs.find((c) => frame >= c.start_frame_index && frame <= c.end_frame_index);
		if (inRange) return inRange;
		return refs
			.slice()
			.sort((a, b) => Math.abs(a.anchor_frame_index - frame) - Math.abs(b.anchor_frame_index - frame))[0] ?? null;
	}

	async function runVlmQuery(question: string) {
		const clip = activeClip;
		if (!clip) {
			scanStatus = 'select an episode before querying VLM';
			return;
		}
		inspectorQuestion = question;
		pushChatTurn({ role: 'user', content: question, ts: Date.now() });
		const req = ++vlmReq;
		vlmBusy = true;
		inspectorAnswer = 'querying…';
		setInspectorActive();
		scanStatus = 'vlm querying…';
		const camera = featuredCamera(clip);
		const currentClipRef = currentEpisodeClipRef(clip);
		const payload = {
			episode_id: clip.episode_id,
			frame_index: currentEpisodeFrame(clip),
			question,
			camera: camera ?? undefined,
			target_clip_id: currentClipRef?.id,
			chat_history: chatHistory.map((m) => ({ role: m.role, content: m.content })),
			max_context_tokens: CHAT_CONTEXT_TOKEN_LIMIT
		};
		try {
			const r = await fetch(`${API}/vlm/query`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(payload)
			});
			const data = await r.json();
			if (!r.ok) throw new Error(data?.detail ?? 'vlm query failed');
			if (req !== vlmReq) return;
			lastVlmResponse = data as VlmQueryResponse;
			inspectorAnswer = `${data.result.incident_type}: ${data.result.summary}`;
			scanStatus = `vlm · ${data.result.incident_type} (${Math.round(data.result.confidence * 100)}%)`;
			pushChatTurn({
				role: 'assistant',
				content: `${data.result.incident_type}: ${data.result.summary}`,
				ts: Date.now()
			});
			setInspectorActive();
		} catch (err) {
			if (req !== vlmReq) return;
			inspectorAnswer = `error: ${err instanceof Error ? err.message : 'query failed'}`;
			scanStatus = `vlm ✕ ${err instanceof Error ? err.message : 'query failed'}`;
			setInspectorActive();
		} finally {
			if (req === vlmReq) vlmBusy = false;
		}
	}

	let activeClip = $derived(activeEp !== null ? clips.get(activeEp) : null);
	let episodeSidebarCards = $derived(
		[...clips.values()]
			.reverse()
			.map(
				(clip): EpisodeSidebarCard => ({
					episode_id: clip.episode_id,
					thumbnail: episodeRepresentativeUrl(clip),
					incidents: incidentGroups(clip).length,
					flagged: clip.moments.length,
					scanning: isScanActive() && currentEpisode === clip.episode_id
				})
			)
	);

	// Auto-scroll chat to bottom on new messages or VLM busy state
	$effect(() => {
		chatHistory; vlmBusy;
		if (viewportMode !== 'chat') return;
		const el = document.getElementById('chat-log-scroll');
		if (el) requestAnimationFrame(() => { el.scrollTop = el.scrollHeight; });
	});
</script>

<div class="page-layout" class:glass-panel={viewportMode === 'main'}>

	<!-- ── Controls bar ── -->
	{#if viewportMode === 'main'}
		<TopBar
			{jobRunning}
			clipsSize={clips.size}
			{scanStatus}
			{scanning}
			{scanPhase}
			{currentEpisode}
			{liveStatus}
			{vlmBusy}
			{lastVlmResponse}
			{exportMsg}
			onScan={onScanButton}
			onLabel={startLabel}
			onExport={doExport}
			{scanButtonLabel}
		/>
	{/if}

	<!-- ── Body ── -->
	<div class="body" class:body-chat={viewportMode === 'chat'}>
	{#if viewportMode === 'main'}

		<!-- Main: camera or detail -->
		<div class="main-col">

			{#if selMoment && activeClip}
				{@const m = selMoment}
				{@const desc = activeClip.descriptions[String(m.cluster_id)]}
				{@const key = `${m.episode_id}-${m.frame_index}`}

				<!-- Frame detail view -->
				<div class="detail-wrap">
					<div class="detail-head">
						<button class="back-btn-inline" onclick={() => selMoment = null}>← back</button>
						<div style="display:flex; align-items:center; gap:8px;">
							<div class="frame-chip-inline" style="background:{clusterColor(m.cluster_id)}">
								{m.cluster_id === -1 ? 'noise' : `cluster ${m.cluster_id}`}
							</div>
							<button class="back-btn-inline" onclick={() => (showMomentInfo = !showMomentInfo)}>
								{showMomentInfo ? 'hide info' : 'details'}
							</button>
						</div>
					</div>
					{#if detailViewsLoading && detailViews.length === 0}
						<p class="detail-loading">loading camera angles…</p>
					{/if}
					<div class="detail-views-grid">
						{#each detailViews as view, i (view.frame_url)}
							<div class="detail-frame-box" class:last-odd={detailViews.length % 2 === 1 && i === detailViews.length - 1}>
								<img src={view.frame_url} alt="" class="detail-frame" />
								<div class="camera-chip">{cameraLabel(view.camera)}</div>
							</div>
						{/each}
					</div>

					{#if m.context_window && m.context_window.length}
						<div class="ctx-strip">
							{#each m.context_window as fi (fi)}
								<button class="ctx-cell" class:ctx-active={fi === m.frame_index}
									onclick={() => selMoment = {...m, frame_index: fi}}>
									<img src={FURL(m.episode_id, fi)} alt="" class="ctx-img"
										onerror={(e)=>{ (e.target as HTMLImageElement).style.opacity='0.15'; }} />
									<span class="ctx-fi">{fi}</span>
								</button>
							{/each}
						</div>
					{/if}

					{#if showMomentInfo}
						<div class="detail-meta">
							<div class="meta-row"><span>episode</span><span>#{m.episode_id}</span></div>
							<div class="meta-row"><span>frame</span><span>{m.frame_index}</span></div>
							{#if desc}<div class="meta-row"><span>description</span><span class="meta-desc">{desc}</span></div>{/if}
						</div>
					{/if}

					<div class="detail-actions">
						<button class="action approve" disabled={validating[key]} onclick={()=>validate(m,'approved')}>approve</button>
						<button class="action reject"  disabled={validating[key]} onclick={()=>validate(m,'rejected')}>reject</button>
					</div>
				</div>

			{:else if activeClip && (!scanning || scanPaused)}
				<div class="waterfall-wrap">
					<div class="waterfall-head">
						<div class="waterfall-head-left">
							<span>episode #{activeClip.episode_id}</span>
							<span>{incidentGroups(activeClip).length} incidents · {activeClip.moments.length} flagged moments</span>
						</div>
						{#if episodeCameraEntries(activeClip).length > 0}
							<div class="head-controls">
								<button class="video-btn" onclick={toggleVideoPlay}>{videoPlaying ? 'pause' : 'play'}</button>
								<button class="video-btn icon-only" aria-label={`Rewind ${adaptiveSeekStepSec().toFixed(1)} seconds`} title={`rewind ${adaptiveSeekStepSec().toFixed(1)}s`} onclick={() => seekVideo(-adaptiveSeekStepSec())}>
									<svg class="video-icon" viewBox="0 0 24 24" aria-hidden="true">
										<path d="M12 5a7 7 0 1 0 6.35 9.95"></path>
										<polyline points="12,5 7.8,5 9.8,7.2"></polyline>
									</svg>
								</button>
								<button class="video-btn icon-only" aria-label={`Forward ${adaptiveSeekStepSec().toFixed(1)} seconds`} title={`forward ${adaptiveSeekStepSec().toFixed(1)}s`} onclick={() => seekVideo(adaptiveSeekStepSec())}>
									<svg class="video-icon" viewBox="0 0 24 24" aria-hidden="true">
										<path d="M12 5a7 7 0 1 1-6.35 9.95"></path>
										<polyline points="12,5 16.2,5 14.2,7.2"></polyline>
									</svg>
								</button>
								<button class="video-btn icon-only" aria-label="Toggle loop" title="loop" onclick={toggleVideoLoop} class:active={videoLoop}>
									<svg class="video-icon" viewBox="0 0 24 24" aria-hidden="true">
										<path d="M4 7h11l-2-2"></path>
										<path d="M20 17H9l2 2"></path>
										<path d="M4 7v4"></path>
										<path d="M20 17v-4"></path>
									</svg>
								</button>
								<button class="video-btn icon-only" aria-label={videoMuted ? 'Unmute' : 'Mute'} title={videoMuted ? 'unmute' : 'mute'} onclick={toggleVideoMute}>
									{#if videoMuted}
										<svg class="video-icon" viewBox="0 0 24 24" aria-hidden="true">
											<polygon points="4,10 8,10 13,6 13,18 8,14 4,14"></polygon>
											<line x1="17" y1="9" x2="21" y2="15"></line>
											<line x1="21" y1="9" x2="17" y2="15"></line>
										</svg>
									{:else}
										<svg class="video-icon" viewBox="0 0 24 24" aria-hidden="true">
											<polygon points="4,10 8,10 13,6 13,18 8,14 4,14"></polygon>
											<path d="M16 10c1.5 1.5 1.5 2.5 0 4"></path>
											<path d="M18.6 8c3.2 3.2 3.2 6.8 0 10"></path>
										</svg>
									{/if}
								</button>
								<button class="video-btn" onclick={cycleVideoRate}>{videoRate.toFixed(2)}x</button>
							</div>
						{/if}
					</div>
						<div class="episode-player">
							<!-- All cameras in DOM for sync, only featured is visible -->
							<div class="episode-camera-wrap">
								{#if episodeCameraEntries(activeClip).length > 0}
									{#each episodeCameraEntries(activeClip) as [camera, url] (camera)}
										<div class="episode-camera-slot" class:camera-hidden={featuredCamera(activeClip) !== camera}>
											<!-- svelte-ignore a11y_media_has_caption -->
											<video
												use:registerCameraVideo={camera}
												src={url}
												class="episode-video-frame"
												preload="metadata"
												playsinline
												onplay={() => { if (episodeVideoEl === cameraVideoEls[camera]) startVideoAnimationLoop(); syncVideoUiState(); }}
												onpause={() => { if (episodeVideoEl === cameraVideoEls[camera]) stopVideoAnimationLoop(); syncVideoUiState(); }}
												onseeking={() => {
													if (episodeVideoEl !== cameraVideoEls[camera]) return;
													if (!episodeVideoEl) return;
													episodeCurrentTime = episodeVideoEl.currentTime;
												}}
												onvolumechange={syncVideoUiState}
												onratechange={syncVideoUiState}
												ontimeupdate={() => {
													if (episodeVideoEl !== cameraVideoEls[camera]) return;
													if (!episodeVideoEl) return;
													episodeCurrentTime = episodeVideoEl.currentTime;
												}}
												onloadedmetadata={() => {
													const el = cameraVideoEls[camera];
													if (!el) return;
													if (episodeVideoEl === el) {
														episodeDuration = el.duration || 0;
														episodeCurrentTime = el.currentTime || 0;
														syncVideoUiState();
													}
												}}
											></video>
										</div>
									{/each}
								{:else}
									<div class="episode-camera-slot">
										<img src={FURL(activeClip.episode_id, currentEpisodeFrame(activeClip))} alt="" class="episode-video-frame" />
									</div>
								{/if}
								<div class="score-overlay">
									{#if smoothedScoreSeries(activeClip).length > 0}
										<Plot
											height={72}
											axes={false}
											grid={false}
											frame={false}
											margin={{ top: 4, right: 4, bottom: 4, left: 4 }}
											x={{ domain: [episodeBounds(activeClip).min, episodeBounds(activeClip).max] }}
											y={{ domain: [0, 1], clamp: true }}
										>
											<AreaY
												data={playedSmoothedSeries(activeClip)}
												x="frame_index"
												y="score"
												fill="rgba(236, 244, 255, 0.17)"
												curve="catmull-rom"
											/>
											<AreaY
												data={flaggedAreaSeries(activeClip)}
												x="frame_index"
												y="score"
												fill="rgba(245, 250, 255, 0.12)"
												curve="step"
											/>
											<Line
												data={smoothedScoreSeries(activeClip)}
												x="frame_index"
												y="score"
												stroke="rgba(247, 250, 255, 0.62)"
												strokeWidth={1.1}
												curve="catmull-rom"
											/>
										</Plot>
										<div
											class="score-timeline"
											role="button"
											tabindex="0"
											aria-label="Scrub timeline"
											onpointerdown={(e) => onTimelinePointerDown(activeClip, e)}
											onpointermove={(e) => onTimelinePointerMove(activeClip, e)}
											onpointerup={onTimelinePointerUp}
											onpointercancel={onTimelinePointerUp}
										>
											<div class="score-timeline-played" style={`width:${playheadPercent(activeClip)}%`}></div>
											<div class="score-timeline-head" style={`left:${playheadPercent(activeClip)}%`}></div>
										</div>
									{:else if episodeScoresLoading}
										<p class="score-plot-empty">loading score trace…</p>
									{:else}
										<p class="score-plot-empty">no score trace yet</p>
									{/if}
									<div class="episode-video-meta">
										<span>frame {currentEpisodeFrame(activeClip)}</span>
									</div>
								</div>
							</div>

							<!-- Camera selector tabs -->
							{#if episodeCameraEntries(activeClip).length > 1}
								<div class="camera-tabs">
									{#each episodeCameraEntries(activeClip) as [camera] (camera)}
										<button
											class="camera-tab"
											class:camera-tab-active={featuredCamera(activeClip) === camera}
											onclick={() => selectFeaturedCamera(activeClip, camera)}
										>{cameraLabel(camera)}</button>
									{/each}
								</div>
							{/if}
						</div>
					</div>
				{:else}
				<!-- Camera view -->
					<div class="camera-wrap">
						{#if primaryScanFrameUrl()}
							<img
								src={primaryScanFrameUrl()}
							alt=""
							class="camera-frame"
						/>
						{#if scanning}<div class="rec-dot"></div>{/if}
						{#if scanning && Object.keys(scanFrameDisplay).length > 1}
							<div class="scan-camera-overlays">
								{#each Object.entries(scanFrameDisplay).slice(1) as [camera, frameUrl] (camera)}
									{#if frameUrl}
										<div class="scan-camera-overlay">
											<img
												src={frameUrl.startsWith('http') ? frameUrl : `http://localhost:8000${frameUrl}`}
												alt=""
												class="scan-camera-video"
											/>
											<span class="episode-overlay episode-camera-label scan-camera-label">{cameraLabel(camera)}</span>
										</div>
									{/if}
								{/each}
								</div>
							{/if}
						{:else if primaryScanVideoUrl()}
							<!-- svelte-ignore a11y_media_has_caption -->
							<video
								src={primaryScanVideoUrl()}
								class="camera-frame"
								autoplay
								muted
								loop
								playsinline
								preload="metadata"
							></video>
							{#if scanning}<div class="rec-dot"></div>{/if}
							{#if scanning && scanOverlayVideoEntries().length > 0}
								<div class="scan-camera-overlays">
									{#each scanOverlayVideoEntries() as [camera, url] (camera)}
										<div class="scan-camera-overlay">
											<!-- svelte-ignore a11y_media_has_caption -->
											<video
												src={url}
												class="scan-camera-video"
												autoplay
												muted
												loop
												playsinline
												preload="metadata"
											></video>
											<span class="episode-overlay episode-camera-label scan-camera-label">{cameraLabel(camera)}</span>
										</div>
									{/each}
								</div>
							{/if}
						{:else if activeClip && activeClip.flagged.length > 0}
							<img src={activeClip.flagged[0].frame_url} alt="" class="camera-frame" />
						{:else}
						<div class="camera-empty">
							<svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
								<path d="M15 10l4.553-2.069A1 1 0 0121 8.867V15.133a1 1 0 01-1.447.902L15 14M3 8a2 2 0 012-2h10a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2V8z"/>
							</svg>
							<p>{clips.size === 0 ? 'run scan to start' : 'select an episode'}</p>
						</div>
					{/if}

					{#if scanning || frameBuf.length > 0 || scanFrameGroups.length > 0}
						<div class="fps-bar">
							<span class="fps-label">{SCAN_PLAYBACK_FPS} fps</span>
							<span class="buf-label">{scanFrameGroups.length || frameBuf.length} buffered</span>
						</div>
					{/if}
				</div>
			{/if}

		</div>

		<!-- ── Clips panel ── -->
		<EpisodesSidebar
			cards={episodeSidebarCards}
			{activeEp}
			onSelectEpisode={(episodeId) => {
				activeEp = episodeId;
				selMoment = null;
			}}
		/>

	{:else}
		<ChatLog {chatHistory} {vlmBusy} />
	{/if}
	</div>

	{#if inspectorOpen && (inspectorQuestion || inspectorAnswer)}
		<div class="inspector-sheet" class:expanded={inspectorExpanded}>
			<button class="inspector-head" onclick={toggleInspectorExpanded} aria-label="Toggle insight sheet">
				<span>insight</span>
				<span>
					{#if vlmBusy}working…{:else if inspectorExpanded}collapse{:else}expand{/if}
				</span>
			</button>
			<div class="inspector-body">
				<p class="inspector-question">{inspectorQuestion || '—'}</p>

				{#if lastVlmResponse?.resolved_clip}
					{@const rc = lastVlmResponse.resolved_clip}
					{@const total = lastVlmResponse.episode_clips?.length ?? 1}
					<p class="inspector-clip-ref">
						clip {rc.ordinal}/{total} · frames {rc.start_frame_index}–{rc.end_frame_index}
						{#if rc.start_timestamp_s != null && rc.end_timestamp_s != null}
							· {rc.start_timestamp_s.toFixed(1)}s–{rc.end_timestamp_s.toFixed(1)}s
						{/if}
					</p>
				{/if}

				<p class="inspector-answer">{inspectorAnswer || '—'}</p>

				{#if lastVlmResponse}
					<div class="inspector-meta">
						<span class="conf-badge" style="opacity:{0.4 + lastVlmResponse.result.confidence * 0.6}">
							{Math.round(lastVlmResponse.result.confidence * 100)}% conf
						</span>
						<span>{lastVlmResponse.result.failure_mode}</span>
						{#if lastVlmResponse.token_usage}
							<span class="tok-count">+{lastVlmResponse.token_usage.total_tokens} tok</span>
						{/if}
					</div>

					{#if lastVlmResponse.result.actionability}
						<p class="inspector-actionability">{lastVlmResponse.result.actionability}</p>
					{/if}

					{#if lastVlmResponse.result.recommendations?.length}
						<ul class="inspector-recs">
							{#each lastVlmResponse.result.recommendations as rec}
								<li>{rec}</li>
							{/each}
						</ul>
					{/if}

					{#if (lastVlmResponse.episode_clips?.length ?? 0) > 1}
						<div class="inspector-clip-nav">
							{#each (lastVlmResponse.episode_clips ?? []) as ec}
								<button
									class="clip-nav-chip"
									class:clip-nav-active={lastVlmResponse.resolved_clip?.id === ec.id}
									onclick={() => {
										if (activeClip) {
											selMoment = activeClip.moments.find(m => m.frame_index >= ec.start_frame_index && m.frame_index <= ec.end_frame_index) ?? activeClip.moments[0] ?? null;
										}
									}}
								>clip {ec.ordinal}</button>
							{/each}
						</div>
					{/if}
				{/if}

				{#if Math.floor(chatHistory.length / 2) > 1}
					<p class="inspector-history">{Math.floor(chatHistory.length / 2)} exchanges in context{#if lastVlmResponse?.context?.history_compacted} · compacted{/if}</p>
				{/if}
			</div>
		</div>
	{/if}
</div>

<style>
	/* ── Layout ── */
	.page-layout { flex:1; display:flex; flex-direction:column; overflow:hidden; min-height:0; position:relative; }

	.body { flex:1; display:flex; overflow:hidden; min-height:0; padding:14px 16px 16px; gap:14px; }
	.body.body-chat { padding:0; gap:0; }

	/* ── Main col ── */
	.main-col { flex:1; min-width:0; display:flex; flex-direction:column; }

	/* ── Camera ── */
	.camera-wrap {
		flex:1; position:relative; border-radius:20px; overflow:hidden;
		background:rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.07);
		display:flex; align-items:center; justify-content:center;
	}
	.camera-frame { width:100%; height:100%; object-fit:cover; display:block; }
	.scan-camera-overlays {
		position:absolute;
		left:14px;
		top:14px;
		display:flex;
		flex-direction:column;
		gap:8px;
		width:min(22%, 220px);
		z-index:4;
		pointer-events:none;
	}
	.scan-camera-overlay {
		position:relative;
		aspect-ratio:16 / 10;
		border-radius:12px;
		overflow:hidden;
		border:1px solid rgba(255,255,255,0.08);
		background:#000;
	}
	.scan-camera-video { width:100%; height:100%; object-fit:cover; display:block; }
	.scan-camera-label {
		left:8px;
		bottom:8px;
		font-size:10px;
	}
	.rec-dot {
		position:absolute; top:14px; right:14px;
		width:8px; height:8px; border-radius:50%;
		background:rgba(255,60,60,0.95);
		box-shadow:0 0 0 0 rgba(255,60,60,0.4);
		animation:pulse 1.4s ease infinite;
	}
	@keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(255,60,60,0.5)} 70%{box-shadow:0 0 0 10px rgba(255,60,60,0)} 100%{box-shadow:0 0 0 0 rgba(255,60,60,0)} }
	.camera-empty { display:flex; flex-direction:column; align-items:center; gap:10px; color:rgba(255,255,255,0.15); }
	.camera-empty p { font-size:13px; margin:0; }

	/* FPS bar */
	.fps-bar {
		position:absolute; bottom:14px; left:50%; transform:translateX(-50%);
		display:flex; align-items:center; gap:10px;
		background:rgba(0,0,0,0.55); backdrop-filter:blur(12px);
		border:1px solid rgba(255,255,255,0.08); border-radius:999px;
		padding:6px 16px;
	}
	.fps-label, .buf-label { font-size:11px; color:rgba(255,255,255,0.45); white-space:nowrap; font-family:monospace; }

	/* ── Detail view ── */
	.detail-wrap { flex:1; overflow-y:auto; display:flex; flex-direction:column; gap:14px; padding-right:4px; }
	.detail-head { display:flex; align-items:center; justify-content:space-between; gap:8px; }
	.detail-views-grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px; }
	.detail-frame-box { position:relative; border-radius:18px; overflow:hidden; border:1px solid rgba(255,255,255,0.08); }
	.detail-frame-box.last-odd { grid-column:1 / -1; width:calc(50% - 5px); justify-self:center; }
	.detail-frame { width:100%; aspect-ratio:4/3; object-fit:cover; display:block; background:rgba(255,255,255,0.03); }
	.frame-chip-inline {
		font-size:11px; padding:3px 10px; border-radius:999px;
		border:1px solid rgba(255,255,255,0.12); color:rgba(255,255,255,0.8);
		backdrop-filter:blur(10px);
	}
	.camera-chip {
		position:absolute; bottom:10px; right:10px;
		font-size:10px; padding:2px 8px; border-radius:999px;
		border:1px solid rgba(255,255,255,0.12); color:rgba(255,255,255,0.8);
		background:rgba(0,0,0,0.45); backdrop-filter:blur(10px);
		text-transform:capitalize;
	}
	.back-btn-inline {
		font-size:11px; padding:4px 10px; border-radius:999px;
		border:1px solid rgba(255,255,255,0.12); color:rgba(255,255,255,0.6);
		background:rgba(0,0,0,0.45); backdrop-filter:blur(10px);
		cursor:pointer; font-family:inherit;
	}
	.back-btn-inline:hover { color:rgba(255,255,255,0.9); }
	.detail-loading { margin:0; font-size:11px; color:rgba(255,255,255,0.45); font-family:monospace; }

	/* Episode waterfall list */
	.waterfall-wrap {
		flex:1;
		min-height:0;
		overflow:hidden;
		display:flex;
		flex-direction:column;
		border-radius:16px;
		border:1px solid rgba(255,255,255,0.08);
		background:rgba(255,255,255,0.02);
		padding:12px;
	}
	.episode-player {
		flex:1;
		min-height:0;
		display:flex;
		flex-direction:column;
		gap:8px;
		padding:10px;
		border-radius:12px;
		border:1px solid rgba(255,255,255,0.08);
		background:rgba(0,0,0,0.28);
		overflow:hidden;
	}
	/* Single-camera wrap — all videos in DOM for sync, only featured shown */
	.episode-camera-wrap {
		position:relative;
		flex:1;
		min-height:0;
		height:min(56vh, calc(100dvh - 340px));
		border-radius:10px;
		overflow:hidden;
		background:#000;
	}
	.episode-camera-slot {
		position:absolute;
		inset:0;
		width:100%;
		height:100%;
	}
	.episode-camera-slot.camera-hidden {
		display:none;
	}
	.episode-video-frame {
		width:100%; height:100%; object-fit:contain; display:block;
		background:#000;
	}

	/* Camera tab selector */
	.camera-tabs {
		display:flex;
		flex-wrap:wrap;
		gap:4px;
		padding:0 2px;
	}
	.camera-tab {
		background:transparent;
		border:1px solid rgba(255,255,255,0.1);
		border-radius:6px;
		color:rgba(255,255,255,0.42);
		font:inherit;
		font-size:11px;
		letter-spacing:0.01em;
		padding:3px 9px;
		cursor:pointer;
		transition:color 0.12s, border-color 0.12s, background 0.12s;
	}
	.camera-tab:hover { color:rgba(255,255,255,0.72); border-color:rgba(255,255,255,0.18); }
	.camera-tab.camera-tab-active {
		color:rgba(255,255,255,0.88);
		border-color:rgba(255,255,255,0.24);
		background:rgba(255,255,255,0.05);
	}
	.score-overlay {
		position:absolute;
		left:0;
		right:0;
		bottom:0;
		padding:48px 10px 8px;
		background:linear-gradient(180deg, rgba(0,0,0,0) 0%, rgba(0,0,0,0.28) 28%, rgba(0,0,0,0.56) 100%);
		pointer-events:none;
	}
	.score-overlay .score-timeline { pointer-events:auto; }
	:global(.score-overlay figure) { margin:0; }
	:global(.score-overlay svg) { width:100%; display:block; }
	.score-timeline {
		position:relative;
		height:6px;
		margin:0 6px 4px;
		border-radius:999px;
		background:rgba(255,255,255,0.14);
		overflow:visible;
		pointer-events:auto;
		cursor:pointer;
		touch-action:none;
	}
	.score-timeline-played {
		position:absolute;
		left:0; top:0; bottom:0;
		border-radius:999px;
		background:linear-gradient(90deg, rgba(255,255,255,0.38), rgba(255,255,255,0.58));
		box-shadow:0 0 10px rgba(255,255,255,0.18);
	}
	.score-timeline-head {
		position:absolute;
		top:50%;
		width:8px;
		height:8px;
		border-radius:50%;
		transform:translate(-50%,-50%);
		background:rgba(180,235,255,1);
		border:1px solid rgba(255,255,255,0.95);
		box-shadow:0 0 0 2px rgba(180,235,255,0.18);
	}
	.episode-video-meta {
		display:flex; align-items:center; justify-content:space-between;
		font-size:10px; color:rgba(255,255,255,0.4);
		padding:2px 6px 0;
		font-family:monospace;
	}
	.video-btn {
		border:none;
		background:transparent;
		color:rgba(255,255,255,0.62);
		padding:0;
		font-size:10px;
		letter-spacing:0.02em;
		font-family:inherit;
		cursor:pointer;
	}
	.video-btn.icon-only {
		display:inline-flex;
		align-items:center;
		justify-content:center;
		min-width:1.35em;
	}
	.video-icon {
		width:1.3em;
		height:1.3em;
		fill:none;
		stroke:currentColor;
		stroke-width:1.9;
		stroke-linecap:round;
		stroke-linejoin:round;
	}
	.video-btn:hover { color:rgba(255,255,255,0.9); }
	.video-btn.active {
		color:rgba(255,255,255,0.98);
	}
	.score-plot-empty {
		margin:0;
		height:70px;
		display:flex;
		align-items:center;
		justify-content:center;
		font-size:10px;
		color:rgba(255,255,255,0.42);
	}
	.waterfall-head {
		display:flex; justify-content:space-between; align-items:center; gap:12px;
		font-size:12px; color:rgba(255,255,255,0.72);
		padding-bottom:10px; margin-bottom:10px;
		flex-shrink:0;
		border-bottom:1px solid rgba(255,255,255,0.08);
	}
	.waterfall-head-left {
		display:flex;
		align-items:center;
		gap:12px;
		min-width:0;
	}
	.head-controls {
		display:flex;
		align-items:center;
		gap:8px;
		flex-wrap:wrap;
		justify-content:flex-end;
	}
	/* Context strip */
	.ctx-strip { display:flex; gap:6px; overflow-x:auto; scrollbar-width:none; }
	.ctx-strip::-webkit-scrollbar { display:none; }
	.ctx-cell { flex-shrink:0; display:flex; flex-direction:column; align-items:center; gap:3px; border:none; background:transparent; cursor:pointer; padding:0; }
	.ctx-img { width:54px; height:40px; object-fit:cover; border-radius:6px; border:1px solid rgba(255,255,255,0.07); transition:border-color 0.12s; }
	.ctx-cell:hover .ctx-img, .ctx-cell.ctx-active .ctx-img { border-color:rgba(255,255,255,0.45); }
	.ctx-fi { font-size:9px; color:rgba(255,255,255,0.22); font-variant-numeric:tabular-nums; }

	/* Metadata */
	.detail-meta { display:flex; flex-direction:column; gap:7px; }
	.meta-row { display:flex; justify-content:space-between; align-items:baseline; gap:12px; font-size:12px; }
	.meta-row span:first-child { color:rgba(255,255,255,0.28); }
	.meta-row span:last-child  { color:rgba(255,255,255,0.75); font-variant-numeric:tabular-nums; text-align:right; }
	.meta-desc { font-size:12px; text-align:right; color:rgba(255,255,255,0.75) !important; }
	.detail-actions { display:flex; gap:8px; }
	.action { flex:1; padding:8px 0; border-radius:10px; border:1px solid rgba(255,255,255,0.08); font-size:13px; font-family:inherit; cursor:pointer; background:transparent; transition:background 0.12s, opacity 0.12s; }
	.action.approve { color:rgba(120,220,140,0.85); border-color:rgba(120,220,140,0.2); }
	.action.approve:hover:not(:disabled) { background:rgba(120,220,140,0.1); }
	.action.reject  { color:rgba(255,100,100,0.8);  border-color:rgba(255,100,100,0.18); }
	.action.reject:hover:not(:disabled)  { background:rgba(255,100,100,0.09); }
	.action:disabled { opacity:0.3; cursor:not-allowed; }

	.episode-overlay {
		position:absolute;
		color:#fff;
		mix-blend-mode:difference;
		pointer-events:none;
		user-select:none;
	}

	@media (max-width: 820px) {
		.detail-views-grid { grid-template-columns:1fr; }
		.detail-frame-box.last-odd { grid-column:auto; width:100%; }
	}

	.inspector-sheet {
		position:absolute;
		left:50%;
		bottom:12px;
		transform:translateX(-50%) translateY(12px);
		width:min(760px, calc(100% - 28px));
		border:1px solid rgba(255,255,255,0.14);
		background:
			linear-gradient(180deg, rgba(255,255,255,0.14) 0%, rgba(255,255,255,0.06) 100%),
			linear-gradient(120deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 46%, rgba(255,255,255,0.05) 100%);
		border-radius:14px;
		z-index:16;
		opacity:0.95;
		transition:transform 0.2s ease, opacity 0.2s ease, border-color 0.2s ease;
		backdrop-filter:blur(34px) saturate(170%);
		-webkit-backdrop-filter:blur(34px) saturate(170%);
		box-shadow:
			0 18px 40px rgba(0,0,0,0.26),
			inset 0 1px 0 rgba(255,255,255,0.22),
			inset 0 -1px 0 rgba(255,255,255,0.06);
	}
	.inspector-sheet.expanded {
		transform:translateX(-50%) translateY(0);
		opacity:1;
		border-color:rgba(255,255,255,0.2);
	}
	.inspector-head {
		width:100%;
		display:flex;
		align-items:center;
		justify-content:space-between;
		gap:12px;
		padding:8px 12px;
		border:none;
		background:transparent;
		color:rgba(255,255,255,0.78);
		font-size:11px;
		letter-spacing:0.05em;
		text-transform:lowercase;
		cursor:pointer;
		border-bottom:1px solid rgba(255,255,255,0.08);
	}
	.inspector-body {
		padding:0 12px 10px;
		display:grid;
		gap:6px;
		max-height:0;
		overflow:hidden;
		transition:max-height 0.2s ease;
	}
	.inspector-sheet.expanded .inspector-body { max-height:220px; }
	.inspector-question {
		margin:0;
		font-size:11px;
		color:rgba(255,255,255,0.66);
		white-space:nowrap;
		overflow:hidden;
		text-overflow:ellipsis;
	}
	.inspector-answer {
		margin:0;
		font-size:12px;
		color:rgba(255,255,255,0.9);
		line-height:1.3;
		line-clamp:2;
		display:-webkit-box;
		-webkit-line-clamp:2;
		-webkit-box-orient:vertical;
		overflow:hidden;
	}
	.inspector-meta {
		display:flex;
		align-items:center;
		gap:10px;
		font-size:10px;
		color:rgba(255,255,255,0.7);
		font-family:monospace;
	}
	.inspector-clip-ref {
		margin:0;
		font-size:10px;
		color:rgba(140,180,255,0.75);
		font-family:monospace;
	}
	.inspector-actionability {
		margin:0;
		font-size:11px;
		color:rgba(255,255,255,0.5);
		font-style:italic;
	}
	.inspector-recs {
		margin:2px 0 0 0;
		padding-left:14px;
		font-size:11px;
		color:rgba(255,255,255,0.65);
		line-height:1.5;
	}
	.inspector-recs li { margin:0; }
	.inspector-clip-nav {
		display:flex;
		gap:4px;
		flex-wrap:wrap;
	}
	.clip-nav-chip {
		padding:2px 8px;
		border-radius:999px;
		border:1px solid rgba(255,255,255,0.1);
		background:transparent;
		color:rgba(255,255,255,0.4);
		font-size:10px;
		font-family:inherit;
		cursor:pointer;
		transition:background 0.1s, color 0.1s;
	}
	.clip-nav-chip:hover { background:rgba(255,255,255,0.06); color:rgba(255,255,255,0.75); }
	.clip-nav-chip.clip-nav-active { background:rgba(255,255,255,0.1); color:rgba(255,255,255,0.9); border-color:rgba(255,255,255,0.22); }
	.conf-badge { font-family:monospace; }
	.tok-count { color:rgba(255,255,255,0.3); }
	.inspector-history {
		margin:0;
		font-size:10px;
		color:rgba(255,255,255,0.35);
	}
</style>
