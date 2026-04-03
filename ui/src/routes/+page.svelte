<script lang="ts">
	import { onMount } from 'svelte';
	import { AreaY, Line, Plot } from 'svelteplot';
	import TopBar from '$lib/components/TopBar.svelte';
	import EpisodesSidebar from '$lib/components/EpisodesSidebar.svelte';
	import ChatLog from '$lib/components/ChatLog.svelte';

	const API  = 'http://localhost:8000/api';
	const FURL = (ep: number, fi: number) => `${API}/frames/${ep}_${fi}.jpg`;
	const mediaUrl = (raw: string, rev = Date.now()) => {
		const base = raw.startsWith('http') ? raw : `http://localhost:8000${raw}`;
		return `${base}${base.includes('?') ? '&' : '?'}rev=${rev}`;
	};

	// ── Types ─────────────────────────────────────────────────────────────────
	type Moment = {
		episode_id: number; frame_index: number; cluster_id: number;
		cluster_size?: number; cluster_span?: number[]; context_window?: number[]; label: string;
		incident_span?: number[];
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
	type EpisodeScoreState = 'pending' | 'scoring' | 'ready' | 'failed';
	type EpisodeScoreDetail = {
		state: EpisodeScoreState;
		trace: EpisodeScorePoint[];
		error?: string | null;
		updated_at?: number;
	};
	type SseEvent = {
		type: 'episode_frames'|'frame'|'progress'|'done'|'error'|'ping';
		episode_id?: number; frame_index?: number; cluster_id?: number;
		cluster_size?: number; frame_url?: string; frame_urls?: string[];
		incident_span?: number[];
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
		incident: string | null;
		context: string | null;
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
		chat_context?: {
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
	type ChatCitation = {
		episode_id: number;
		frame_index: number;
		frame_start?: number;
		frame_end?: number;
		camera?: string;
		cluster_id?: number;
	};
	type ChatTurn = {
		role: 'user' | 'assistant';
		content: string;
		ts: number;
		incident?: string | null;
		citation?: ChatCitation;
		status?: string;
	};
	type EpisodeSidebarCard = {
		episode_id: number;
		thumbnail: string | null;
		thumbnailKind: 'image' | 'video';
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
		let episodeScoresLoadingByEpisode = $state<Record<number, boolean>>({});
		let episodeScoresReadyByEpisode = $state<Record<number, boolean>>({});
		let episodeScoreStateByEpisode = $state<Record<number, EpisodeScoreState>>({});
		let episodeScoreErrorByEpisode = $state<Record<number, string | null>>({});
		let episodeScoresRetryAfterByEpisode = $state<Record<number, number>>({});
		let episodeScoresRetryNonce = $state<Record<number, number>>({});
		let episodeScoresReq = 0;
		let episodeVideoUrls = $state<Record<number, string>>({});
		let episodeCameraVideoUrls = $state<Record<number, Record<string, string>>>({});
		let episodeMediaLoadingByEpisode = $state<Record<number, boolean>>({});
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
	let catalogLoading = $state(true);
	let viewportMode = $state<'main' | 'chat'>('main');

	// Jobs
	let jobRunning = $state<'scan'|'label'|null>(null);
	let exportMsg  = $state<string|null>(null);

	let es: EventSource|null = null;
	let esReconnectTimer: ReturnType<typeof setTimeout> | null = null;
	let episodeScoreRetryTimers = new Map<number, ReturnType<typeof setTimeout>>();

	function clearEpisodeScoreRetryTimer(episodeId: number) {
		const timer = episodeScoreRetryTimers.get(episodeId);
		if (timer) clearTimeout(timer);
		episodeScoreRetryTimers.delete(episodeId);
	}

	function scheduleEpisodeScoreRetry(episodeId: number, delayMs = 1500) {
		clearEpisodeScoreRetryTimer(episodeId);
		episodeScoresRetryAfterByEpisode = {
			...episodeScoresRetryAfterByEpisode,
			[episodeId]: Date.now() + delayMs
		};
		const timer = setTimeout(() => {
			episodeScoreRetryTimers.delete(episodeId);
			episodeScoresRetryAfterByEpisode = {
				...episodeScoresRetryAfterByEpisode,
				[episodeId]: 0
			};
			episodeScoresRetryNonce = {
				...episodeScoresRetryNonce,
				[episodeId]: Date.now()
			};
		}, delayMs);
		episodeScoreRetryTimers.set(episodeId, timer);
	}

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
			for (const timer of episodeScoreRetryTimers.values()) clearTimeout(timer);
			episodeScoreRetryTimers.clear();
			stopVideoAnimationLoop();
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

	function latestAssistantIncident(): string | null {
		for (let i = chatHistory.length - 1; i >= 0; i -= 1) {
			const turn = chatHistory[i];
			if (turn.role !== 'assistant') continue;
			const value = turn.incident?.trim();
			if (value) return value;
		}
		return null;
	}

	function latestAssistantPreview(): string | null {
		const compact = latestAssistantIncident();
		if (!compact) return null;
		return compact;
	}

	function openLatestChat() {
		viewportMode = 'chat';
		requestAnimationFrame(() => {
			const el = document.getElementById('chat-log-scroll');
			if (el) el.scrollTop = el.scrollHeight;
		});
	}

	async function validateCitation(citation: ChatCitation, verdict: 'approved' | 'rejected') {
		const clip = clips.get(citation.episode_id);
		if (!clip) return;
		const start = citation.frame_start ?? citation.frame_index;
		const end = citation.frame_end ?? citation.frame_index;
		const moment =
			clip.moments.find((m) => m.frame_index === citation.frame_index) ??
			clip.moments.find((m) => m.frame_index >= start && m.frame_index <= end) ??
			clip.moments[0];
		if (!moment) return;
		await validate(moment, verdict);
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
		activeClip;
		episodeCurrentTime;
		episodeDuration;
		scanning;
		if (!activeClip || scanning) return;
		updateEpisodeFrameState();
	});

		$effect(() => {
			const clip = activeClip;
			scanning;
			episodeScoresRetryNonce[clip?.episode_id ?? -1];
			if (!clip) return;
			if (episodeScoresReadyByEpisode[clip.episode_id]) return;
			if (episodeScoresLoadingByEpisode[clip.episode_id]) return;
			if ((episodeScoresRetryAfterByEpisode[clip.episode_id] ?? 0) > Date.now()) return;
			const req = ++episodeScoresReq;
			episodeScoresLoading = true;
			episodeScoresLoadingByEpisode = { ...episodeScoresLoadingByEpisode, [clip.episode_id]: true };
			void (async () => {
				try {
					const r = await fetch(`${API}/episode-scores/${clip.episode_id}/detail`);
					const detail: EpisodeScoreDetail = r.ok
						? await r.json()
						: { state: 'failed', trace: [], error: 'score detail request failed' };
					const rows: EpisodeScorePoint[] = (detail.trace ?? []).sort((a, b) => a.frame_index - b.frame_index);
					if (req === episodeScoresReq) {
						episodeScores = {
							...episodeScores,
							[clip.episode_id]: rows
						};
						episodeScoreStateByEpisode = {
							...episodeScoreStateByEpisode,
							[clip.episode_id]: detail.state
						};
						episodeScoreErrorByEpisode = {
							...episodeScoreErrorByEpisode,
							[clip.episode_id]: detail.error ?? null
						};
						const hasTrace = rows.length > 0;
						const terminal = detail.state === 'ready' || detail.state === 'failed';
						episodeScoresReadyByEpisode = {
							...episodeScoresReadyByEpisode,
							[clip.episode_id]: terminal
						};
						if (terminal) {
							clearEpisodeScoreRetryTimer(clip.episode_id);
							episodeScoresRetryAfterByEpisode = {
								...episodeScoresRetryAfterByEpisode,
								[clip.episode_id]: 0
							};
						} else {
							scheduleEpisodeScoreRetry(clip.episode_id);
						}
					}
				} catch {
					if (req === episodeScoresReq) {
						episodeScores = { ...episodeScores, [clip.episode_id]: [] };
						episodeScoreStateByEpisode = {
							...episodeScoreStateByEpisode,
							[clip.episode_id]: 'failed'
						};
						episodeScoreErrorByEpisode = {
							...episodeScoreErrorByEpisode,
							[clip.episode_id]: 'score detail request failed'
						};
						episodeScoresReadyByEpisode = {
							...episodeScoresReadyByEpisode,
							[clip.episode_id]: true
						};
						clearEpisodeScoreRetryTimer(clip.episode_id);
					}
				} finally {
					if (req === episodeScoresReq) episodeScoresLoading = false;
					episodeScoresLoadingByEpisode = { ...episodeScoresLoadingByEpisode, [clip.episode_id]: false };
				}
			})();
		});

		$effect(() => {
			const clip = activeClip;
			if (!clip) return;
			if (episodeCameraVideoUrls[clip.episode_id]) return;
			const req = ++episodeCameraVideoReq;
			episodeMediaLoadingByEpisode = { ...episodeMediaLoadingByEpisode, [clip.episode_id]: true };
			void (async () => {
				try {
				const r = await fetch(`${API}/episode-videos/${clip.episode_id}`);
				if (!r.ok) return;
				const payload = (await r.json()) as Record<string, string>;
				if (req !== episodeCameraVideoReq) return;
					const full: Record<string, string> = {};
					for (const [camera, raw] of Object.entries(payload ?? {})) {
						full[camera] = mediaUrl(raw, req);
					}
					if (Object.keys(full).length === 0) return;
					episodeCameraVideoUrls = { ...episodeCameraVideoUrls, [clip.episode_id]: full };
				} catch {
					// ignore missing multi-camera videos
				} finally {
					if (req === episodeCameraVideoReq && episodeVideoUrls[clip.episode_id]) {
						episodeMediaLoadingByEpisode = { ...episodeMediaLoadingByEpisode, [clip.episode_id]: false };
					}
				}
			})();
		});

		$effect(() => {
			const clip = activeClip;
			if (!clip) return;
			if (episodeVideoUrls[clip.episode_id]) return;
			const req = ++episodeVideoReq;
			episodeMediaLoadingByEpisode = { ...episodeMediaLoadingByEpisode, [clip.episode_id]: true };
			void (async () => {
				try {
				const r = await fetch(`${API}/episode-video/${clip.episode_id}`);
				if (!r.ok) return;
				const payload = await r.json();
					const raw = payload?.video_url as string | undefined;
					if (!raw || req !== episodeVideoReq) return;
					const full = mediaUrl(raw, req);
					episodeVideoUrls = { ...episodeVideoUrls, [clip.episode_id]: full };
				} catch {
					// ignore missing server video
				} finally {
					if (req === episodeVideoReq) {
						episodeMediaLoadingByEpisode = { ...episodeMediaLoadingByEpisode, [clip.episode_id]: false };
					}
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
		catalogLoading = true;
		try {
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
			for (const episodeId of map.keys()) {
				void hydrateEpisodeMedia(episodeId);
			}
		} finally {
			if (req === catalogLoadReq) catalogLoading = false;
		}
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
							[episodeId]: mediaUrl(raw, episodeId)
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
					mapped[camera] = mediaUrl(raw, episodeId);
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
						video_url: e.video_url ? mediaUrl(e.video_url, e.episode_id ?? Date.now()) : undefined,
					});
				clips = new Map(clips);
			}
			if (e.video_url && e.episode_id !== undefined) {
					episodeVideoUrls = {
						...episodeVideoUrls,
						[e.episode_id]: mediaUrl(e.video_url, e.episode_id)
					};
				}
			if (e.camera_videos && e.episode_id !== undefined) {
				const mapped: Record<string, string> = {};
				for (const [camera, raw] of Object.entries(e.camera_videos)) {
					mapped[camera] = mediaUrl(raw, e.episode_id ?? Date.now());
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
				const existingMoment = clip.moments.find((m) => m.frame_index === e.frame_index!);
				if (existingMoment) {
					existingMoment.cluster_id = e.cluster_id!;
					existingMoment.cluster_size = e.cluster_size;
					existingMoment.incident_span = e.incident_span;
				} else {
					clip.moments = [
						...clip.moments,
						{
							episode_id: e.episode_id!,
							frame_index: e.frame_index!,
							cluster_id: e.cluster_id!,
							cluster_size: e.cluster_size,
							cluster_span: [],
							context_window: [],
							incident_span: e.incident_span,
							label: 'anomaly'
						}
					];
				}
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
			frameBuf = [];
			displayFrame = '';
			scanFrameGroups = [];
			scanFrameDisplay = {};
			exportMsg = null;
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
		if (clip.video_url) return clip.video_url;
		const primaryVideo = episodeVideoUrls[clip.episode_id];
		if (primaryVideo) return primaryVideo;
		const cameraVideo = Object.values(episodeCameraVideoUrls[clip.episode_id] ?? {})[0];
		if (cameraVideo) return cameraVideo;
		return clip.preview_urls[0] ?? null;
	}

	function episodeRepresentativeKind(clip: Clip): 'image' | 'video' {
		const url = episodeRepresentativeUrl(clip);
		return url && url.toLowerCase().includes('.mp4') ? 'video' : 'image';
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
		const trace = scoreSeries(clip);
		if (trace.length > 0) {
			const min = trace[0].frame_index;
			const max = trace[trace.length - 1].frame_index;
			return { min, max, span: Math.max(1, max - min) };
		}
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
		const pct = currentPlaybackPercent();
		const targetIndex = Math.max(0, Math.min(series.length - 1, Math.round(pct * (series.length - 1))));
		const frame = series[targetIndex]?.frame_index ?? series[0].frame_index;
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
		const pct = currentPlaybackPercent();
		const frame = series[Math.max(0, Math.min(series.length - 1, Math.round(pct * (series.length - 1))))]?.frame_index
			?? currentEpisodeFrame(clip);
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
		const incidentSegments = clip.moments
			.map((m) => {
				if (Array.isArray(m.incident_span) && m.incident_span.length === 2) {
					return {
						start: Math.min(m.incident_span[0], m.incident_span[1]),
						end: Math.max(m.incident_span[0], m.incident_span[1])
					};
				}
				return null;
			})
			.filter((seg): seg is FlagSegment => seg !== null)
			.sort((a, b) => a.start - b.start);
		if (incidentSegments.length > 0) return incidentSegments;
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

	function currentPlaybackPercent(): number {
		if (episodeDuration > 0) {
			return Math.max(0, Math.min(1, currentPlaybackTime() / episodeDuration));
		}
		return 0;
	}

	function playheadPercent(clip: Clip): number {
		void clip;
		return currentPlaybackPercent() * 100;
	}

	function seekToPercent(clip: Clip, pct: number) {
		const p = Math.max(0, Math.min(1, pct));
		const lead = resolveEpisodeVideoTarget();
		if (lead && episodeDuration > 0) {
			const next = p * episodeDuration;
			lead.currentTime = next;
			episodeCurrentTime = next;
			updateEpisodeFrameState();
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
			clearTimeout(videoRaf);
			videoRaf = null;
		}
	}

	function featuredVideoElement(): HTMLVideoElement | null {
		const clip = activeClip;
		if (!clip) return episodeVideoEl;
		const camera = featuredCamera(clip);
		if (camera && cameraVideoEls[camera]?.isConnected) return cameraVideoEls[camera];
		return episodeVideoEl;
	}

	function startVideoAnimationLoop() {
		const target = featuredVideoElement() ?? resolveEpisodeVideoTarget();
		if (videoAnimating || !target) return;
		episodeVideoEl = target;
		videoAnimating = true;
		const updateFromActive = () => {
			const active = featuredVideoElement() ?? resolveEpisodeVideoTarget();
			if (!videoAnimating || !active) return null;
			episodeVideoEl = active;
			episodeCurrentTime = active.currentTime;
			episodeDuration = Number.isFinite(active.duration) ? active.duration : episodeDuration;
			updateEpisodeFrameState();
			return active;
		};
		const tick = () => {
			const active = updateFromActive();
			if (!active) return;
			if (!active.paused) {
				videoRaf = window.setTimeout(tick, 33) as unknown as number;
				return;
			}
			stopVideoAnimationLoop();
		};
		videoRaf = window.setTimeout(tick, 33) as unknown as number;
	}

		function syncVideoUiState() {
			if (!episodeVideoEl) return;
			videoPlaying = !episodeVideoEl.paused;
			videoMuted = episodeVideoEl.muted;
			videoLoop = episodeVideoEl.loop;
			videoRate = episodeVideoEl.playbackRate;
		}

		function connectedCameraVideoEls(): Record<string, HTMLVideoElement> {
			return Object.fromEntries(
				Object.entries(cameraVideoEls).filter(([, el]) => el && el.isConnected)
			);
		}

		function allEpisodeVideoEls(): HTMLVideoElement[] {
			const connected = Object.values(connectedCameraVideoEls());
			if (connected.length > 0) return connected;
			if (typeof document === 'undefined') return [];
			return Array.from(document.querySelectorAll('.episode-camera-wrap .episode-video-frame'));
		}

		function peekEpisodeVideoTarget(): HTMLVideoElement | null {
			const connected = connectedCameraVideoEls();
			if (episodeVideoEl && episodeVideoEl.isConnected && Object.values(connected).includes(episodeVideoEl)) {
				return episodeVideoEl;
			}
			const clip = activeClip;
			if (clip) {
				const camera = featuredCamera(clip);
				if (camera && connected[camera]) return connected[camera];
			}
			return Object.values(connected)[0] ?? allEpisodeVideoEls()[0] ?? null;
		}

		function resolveEpisodeVideoTarget(): HTMLVideoElement | null {
			const fallback = peekEpisodeVideoTarget();
			if (fallback) {
				episodeVideoEl = fallback;
				episodeDuration = Number.isFinite(fallback.duration) ? fallback.duration : episodeDuration;
				episodeCurrentTime = fallback.currentTime || episodeCurrentTime;
				syncVideoUiState();
			}
			return fallback;
		}

		function currentPlaybackTime(): number {
			// episodeCurrentTime is $state, updated by the animation loop every 33ms.
			// Reading it here makes playheadPercent a reactive dependency so Svelte
			// re-renders the timeline bar as the video plays.
			return episodeCurrentTime;
		}

		function syncFollowerVideos(lead: HTMLVideoElement) {
			for (const el of allEpisodeVideoEls()) {
				if (el === lead) continue;
				el.muted = lead.muted;
				el.loop = lead.loop;
				el.playbackRate = lead.playbackRate;
				const duration = Number.isFinite(el.duration) ? el.duration : lead.duration;
				const targetTime = Math.max(0, Math.min(duration || 0, lead.currentTime));
				if (Math.abs(el.currentTime - targetTime) > 0.08) {
					el.currentTime = targetTime;
				}
			}
		}

		async function toggleVideoPlay() {
			const lead = resolveEpisodeVideoTarget();
			if (!lead) return;
			const els = allEpisodeVideoEls();
			if (lead.paused) {
				try {
					await lead.play();
				} catch {
					return;
				}
				episodeVideoEl = lead;
				episodeCurrentTime = lead.currentTime;
				episodeDuration = Number.isFinite(lead.duration) ? lead.duration : episodeDuration;
				updateEpisodeFrameState();
				startVideoAnimationLoop();
				syncFollowerVideos(lead);
				for (const el of els) {
					if (el === lead) continue;
					void el.play().catch(() => undefined);
				}
			} else {
				for (const el of els) el.pause();
			}
			syncVideoUiState();
		}

		function seekVideo(deltaSec: number) {
			const lead = resolveEpisodeVideoTarget();
			if (!lead) return;
			const els = allEpisodeVideoEls();
			const next = Math.max(0, Math.min(lead.duration || 0, lead.currentTime + deltaSec));
			for (const el of els) {
				const duration = Number.isFinite(el.duration) ? el.duration : lead.duration;
				el.currentTime = Math.max(0, Math.min(duration || 0, next));
			}
		episodeCurrentTime = next;
		updateEpisodeFrameState();
	}

	function adaptiveSeekStepSec(): number {
		const duration = episodeDuration > 0 ? episodeDuration : (episodeVideoEl?.duration ?? 0);
		if (!Number.isFinite(duration) || duration <= 0) return 10;
		return Math.max(1, Math.min(10, duration * 0.1));
	}

		function toggleVideoLoop() {
			const lead = resolveEpisodeVideoTarget();
			if (!lead) return;
			const els = allEpisodeVideoEls();
			const next = !lead.loop;
			for (const el of els) el.loop = next;
			syncVideoUiState();
		}

		function toggleVideoMute() {
			const lead = resolveEpisodeVideoTarget();
			if (!lead) return;
			const els = allEpisodeVideoEls();
			const next = !lead.muted;
			for (const el of els) el.muted = next;
			syncVideoUiState();
		}

		function cycleVideoRate() {
			const lead = resolveEpisodeVideoTarget();
			if (!lead) return;
			const els = allEpisodeVideoEls();
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

		function episodeMediaLoading(clip: Clip): boolean {
			return Boolean(episodeMediaLoadingByEpisode[clip.episode_id]);
		}

		function episodeScoreLoading(clip: Clip): boolean {
			return Boolean(episodeScoresLoadingByEpisode[clip.episode_id]);
		}

		function episodeScoreState(clip: Clip): EpisodeScoreState {
			return episodeScoreStateByEpisode[clip.episode_id] ?? 'pending';
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

		function isFeaturedReviewCamera(camera: string): boolean {
			const clip = activeClip;
			if (!clip) return false;
			return featuredCamera(clip) === camera;
		}

		function selectFeaturedCamera(clip: Clip, camera: string) {
		const previous = resolveEpisodeVideoTarget();
		const wasPlaying = previous ? !previous.paused : false;
		const priorTime = previous?.currentTime ?? episodeCurrentTime;
		const priorMuted = previous?.muted ?? videoMuted;
		const priorLoop = previous?.loop ?? videoLoop;
		const priorRate = previous?.playbackRate ?? videoRate;
		featuredCameraByEpisode = { ...featuredCameraByEpisode, [clip.episode_id]: camera };
		requestAnimationFrame(() => {
			const el = cameraVideoEls[camera];
			if (!el) return;
			el.muted = priorMuted;
			el.loop = priorLoop;
			el.playbackRate = priorRate;
			const duration = Number.isFinite(el.duration) ? el.duration : episodeDuration;
			el.currentTime = Math.max(0, Math.min(duration || priorTime, priorTime));
			episodeVideoEl = el;
			episodeDuration = duration || episodeDuration;
			episodeCurrentTime = el.currentTime;
			updateEpisodeFrameState();
			if (wasPlaying) {
				for (const other of allEpisodeVideoEls()) {
					if (other !== el) other.pause();
				}
				void el.play().catch(() => undefined);
				startVideoAnimationLoop();
			} else {
				syncVideoUiState();
			}
		});
		}

		function bindActiveReviewVideo(camera: string) {
			const el = cameraVideoEls[camera];
			if (!el) return;
			episodeVideoEl = el;
			episodeDuration = Number.isFinite(el.duration) ? el.duration : episodeDuration;
			episodeCurrentTime = el.currentTime || episodeCurrentTime;
			syncVideoUiState();
		}

	function onReviewVideoLoaded(camera: string) {
		if (!isFeaturedReviewCamera(camera)) return;
		bindActiveReviewVideo(camera);
	}

	function onReviewVideoTimeUpdate(camera: string) {
		if (!isFeaturedReviewCamera(camera)) return;
		const el = cameraVideoEls[camera];
			if (!el) return;
		if (episodeVideoEl !== el) episodeVideoEl = el;
		episodeDuration = Number.isFinite(el.duration) ? el.duration : episodeDuration;
		episodeCurrentTime = el.currentTime;
		updateEpisodeFrameState();
		syncVideoUiState();
	}

		function reviewLoadingText(clip: Clip): string | null {
			const scoreState = episodeScoreState(clip);
			if (episodeMediaLoading(clip) && (episodeScoreLoading(clip) || scoreState === 'pending' || scoreState === 'scoring')) return 'loading episode media and score trace…';
			if (episodeMediaLoading(clip)) return 'loading episode media…';
			if (scoreState === 'pending') return 'score trace queued…';
			if (scoreState === 'scoring' || episodeScoreLoading(clip)) return 'scoring trace…';
			return null;
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

		function computeEpisodeFrame(clip: Clip): number {
			const playbackTime = currentPlaybackTime();
			const trace = scoreSeries(clip);
			if (trace.length > 0 && episodeDuration > 0) {
				const pct = Math.max(0, Math.min(1, playbackTime / episodeDuration));
				const played = Math.max(0, Math.min(trace.length - 1, Math.round(pct * (trace.length - 1))));
				return trace[played]?.frame_index ?? trace[0].frame_index;
			}
			const videoUrl = episodeVideoUrls[clip.episode_id] || episodeCameraEntries(clip).length > 0;
			if (videoUrl && episodeDuration > 0) {
				const { min, span } = episodeBounds(clip);
				return Math.round(min + (playbackTime / episodeDuration) * span);
			}
			if (trace.length > 0) {
				const played = Math.max(0, Math.min(trace.length - 1, Math.round((playbackTime || 0) * 8)));
				return trace[played]?.frame_index ?? trace[0].frame_index;
			}
			return episodeFrame ?? selMoment?.frame_index ?? clip.moments[0]?.frame_index ?? 0;
		}

		function updateEpisodeFrameState() {
			const clip = activeClip;
			if (!clip || scanning) return;
			episodeFrame = computeEpisodeFrame(clip);
		}

		function currentEpisodeFrame(clip: Clip): number {
			if (activeClip && clip.episode_id === activeClip.episode_id && episodeFrame !== null) {
				return episodeFrame;
			}
			return computeEpisodeFrame(clip);
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
		pushChatTurn({ role: 'user', content: question, ts: Date.now() });
		const req = ++vlmReq;
		vlmBusy = true;
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
			const incident = data.incident?.trim() || latestAssistantIncident();
			scanStatus = incident ? `vlm · ${incident}` : 'vlm response';
			pushChatTurn({
				role: 'assistant',
				content: data.context ?? '',
				ts: Date.now(),
				incident: data.incident?.trim() || null,
				citation: {
					episode_id: payload.episode_id,
					frame_index: currentClipRef?.anchor_frame_index ?? payload.frame_index,
					frame_start: currentClipRef?.start_frame_index ?? payload.frame_index,
					frame_end: currentClipRef?.end_frame_index ?? payload.frame_index,
					camera: payload.camera,
					cluster_id:
						clip.moments.find((m) =>
							currentClipRef
								? m.frame_index >= currentClipRef.start_frame_index && m.frame_index <= currentClipRef.end_frame_index
								: m.frame_index === payload.frame_index
						)?.cluster_id
				},
				status: currentClipRef
					? `clip ${currentClipRef.ordinal}/${Math.max(episodeClipRefs[clip.episode_id]?.length ?? 1, 1)}`
					: undefined
			});
		} catch (err) {
			if (req !== vlmReq) return;
			scanStatus = `vlm ✕ ${err instanceof Error ? err.message : 'query failed'}`;
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
						thumbnailKind: episodeRepresentativeKind(clip),
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
			latestVlmPreview={latestAssistantPreview()}
			{exportMsg}
			onScan={onScanButton}
			onLabel={startLabel}
			onExport={doExport}
			onOpenLatestChat={openLatestChat}
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
														data-camera={camera}
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
														onReviewVideoTimeUpdate(camera);
													}}
													onloadedmetadata={() => {
														onReviewVideoLoaded(camera);
													}}
													onloadeddata={() => {
														onReviewVideoLoaded(camera);
													}}
												></video>
											</div>
										{/each}
									{:else}
									<div class="episode-camera-slot">
										<img src={FURL(activeClip.episode_id, currentEpisodeFrame(activeClip))} alt="" class="episode-video-frame" />
									</div>
								{/if}
								<!-- Frame number overlay — top-left, updates with playhead -->
								<div class="frame-number-overlay">
									frame {currentEpisodeFrame(activeClip)}
								</div>
								<div class="score-overlay">
										{#if reviewLoadingText(activeClip)}
											<p class="score-plot-empty score-loading">{reviewLoadingText(activeClip)}</p>
										{/if}
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
										{:else if episodeScoreState(activeClip) === 'failed'}
											<p class="score-plot-empty">score trace unavailable</p>
										{:else if episodeScoreLoading(activeClip) || episodeScoreState(activeClip) === 'pending' || episodeScoreState(activeClip) === 'scoring'}
											<p class="score-plot-empty">{reviewLoadingText(activeClip) ?? 'loading score trace…'}</p>
									{:else}
										<p class="score-plot-empty">no score trace yet</p>
									{/if}
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
				loading={catalogLoading}
				onSelectEpisode={(episodeId) => {
				activeEp = episodeId;
				selMoment = null;
			}}
		/>

	{:else}
		<ChatLog {chatHistory} {vlmBusy} onValidateCitation={validateCitation} />
	{/if}
	</div>
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
		border:none;
		background:rgba(0,0,0,0.28);
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
		border:none;
		background:transparent;
		overflow:hidden;
	}
	.frame-number-overlay {
		position:absolute;
		top:10px; left:10px;
		z-index:5;
		font-family:monospace;
		font-size:11px;
		color:rgba(255,255,255,0.9);
		background:rgba(0,0,0,0.52);
		backdrop-filter:blur(4px);
		-webkit-backdrop-filter:blur(4px);
		border:1px solid rgba(255,255,255,0.1);
		border-radius:5px;
		padding:3px 7px;
		pointer-events:none;
		letter-spacing:0.03em;
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
		border:none;
		border-radius:6px;
		color:rgba(255,255,255,0.42);
		font:inherit;
		font-size:11px;
		letter-spacing:0.01em;
		padding:3px 9px;
		cursor:pointer;
		transition:color 0.12s, background 0.12s;
	}
	.camera-tab:hover { color:rgba(255,255,255,0.72); }
	.camera-tab.camera-tab-active {
		color:rgba(255,255,255,0.88);
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
		user-select:none;
		-webkit-user-select:none;
		-webkit-touch-callout:none;
		-webkit-tap-highlight-color:transparent;
	}
	.score-overlay .score-timeline { pointer-events:auto; }
	:global(.score-overlay figure) { margin:0; }
	:global(.score-overlay svg) { width:100%; display:block; }
	:global(.score-overlay *),
	:global(.score-overlay svg *),
	:global(.score-overlay .score-timeline),
	:global(.score-overlay .score-timeline *) {
		user-select:none;
		-webkit-user-select:none;
		-webkit-touch-callout:none;
		-webkit-tap-highlight-color:transparent;
		outline:none;
	}
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
	.score-loading {
		position:absolute;
		left:14px;
		top:14px;
		margin:0;
		max-width:260px;
		text-align:left;
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

</style>
