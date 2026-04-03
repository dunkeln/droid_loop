<script lang="ts">
	import { onDestroy } from 'svelte';

	const API = '/api';

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

	type ClipFramePayload = {
		frame_index: number;
		views: Array<{
			camera: string;
			frame_url: string;
		}>;
	};

	type ClipPreview = {
		episode_id: number;
		frame_index: number;
		frames: ClipFramePayload[];
		byCamera: Array<{
			camera: string;
			frameUrls: string[];
		}>;
	};

	let { chatHistory, vlmBusy, onValidateCitation } = $props<{
		chatHistory: ChatTurn[];
		vlmBusy: boolean;
		onValidateCitation: (citation: ChatCitation, verdict: 'approved' | 'rejected') => Promise<void>;
	}>();

	let openPreviewKey = $state<string | null>(null);
	let previewCache = $state<Record<string, ClipPreview>>({});
	let previewLoading = $state<Record<string, boolean>>({});
	let previewError = $state<Record<string, string | null>>({});
	let validating = $state<Record<string, boolean>>({});
	let previewTick = $state(0);
	let previewTimer: ReturnType<typeof setInterval> | null = null;

	function citationKey(citation: ChatCitation): string {
		return `${citation.episode_id}:${citation.frame_start ?? citation.frame_index}:${citation.frame_end ?? citation.frame_index}`;
	}

	function messagePreviewKey(msg: ChatTurn, idx: number): string {
		return `${msg.ts}-${idx}`;
	}

	function clipStatus(status?: string): string | null {
		if (!status) return null;
		const value = status.trim();
		return value.startsWith('clip ') ? value : null;
	}

	function messageContent(content: string): string | null {
		const value = content.trim();
		return value.length > 0 ? value : null;
	}

	function frameUrl(raw: string): string {
		return raw;
	}

	function cameraLabel(camera: string): string {
		return camera.replaceAll('_', ' ');
	}

	function clusterLabel(citation: ChatCitation): string | null {
		if (citation.cluster_id === undefined || citation.cluster_id === null) return null;
		return citation.cluster_id === -1 ? 'noise' : `cluster ${citation.cluster_id}`;
	}

	function startPreviewTimer() {
		if (previewTimer) return;
		previewTimer = setInterval(() => {
			previewTick += 1;
		}, 180);
	}

	function stopPreviewTimer() {
		if (!previewTimer) return;
		clearInterval(previewTimer);
		previewTimer = null;
	}

	onDestroy(() => {
		stopPreviewTimer();
	});

	async function togglePreview(citation: ChatCitation, messageKey: string) {
		const key = citationKey(citation);
		if (openPreviewKey === messageKey) {
			openPreviewKey = null;
			stopPreviewTimer();
			return;
		}
		openPreviewKey = messageKey;
		startPreviewTimer();
		if (previewCache[key] || previewLoading[key]) return;
		previewLoading = { ...previewLoading, [key]: true };
		previewError = { ...previewError, [key]: null };
		try {
			const r = await fetch(`${API}/clip-frames/${citation.episode_id}/${citation.frame_index}`);
			const payload = r.ok ? await r.json() : null;
			if (!payload?.frames || !Array.isArray(payload.frames)) {
				throw new Error('clip preview unavailable');
			}
			const byCameraMap = new Map<string, string[]>();
			for (const frame of payload.frames as ClipFramePayload[]) {
				for (const view of frame.views ?? []) {
					const camera = view.camera ?? 'primary';
					const urls = byCameraMap.get(camera) ?? [];
					urls.push(frameUrl(view.frame_url));
					byCameraMap.set(camera, urls);
				}
			}
			const byCamera = [...byCameraMap.entries()].map(([camera, frameUrls]) => ({
				camera,
				frameUrls
			}));
			previewCache = {
				...previewCache,
				[key]: {
					episode_id: citation.episode_id,
					frame_index: citation.frame_index,
					frames: payload.frames as ClipFramePayload[],
					byCamera
				}
			};
		} catch (err) {
			previewError = {
				...previewError,
				[key]: err instanceof Error ? err.message : 'clip preview unavailable'
			};
		} finally {
			previewLoading = { ...previewLoading, [key]: false };
		}
	}

	function previewFrameUrl(preview: ClipPreview, camera: string): string | null {
		const group = preview.byCamera.find((item) => item.camera === camera);
		if (!group || group.frameUrls.length === 0) return null;
		return group.frameUrls[previewTick % group.frameUrls.length];
	}

	async function handleValidate(citation: ChatCitation, verdict: 'approved' | 'rejected') {
		const key = citationKey(citation);
		validating = { ...validating, [key]: true };
		try {
			await onValidateCitation(citation, verdict);
		} finally {
			validating = { ...validating, [key]: false };
		}
	}
</script>

<div class="chat-col">
	<div class="chat-log" id="chat-log-scroll">
		{#if chatHistory.length === 0}
			<div class="chat-empty-wrap">
				<p class="chat-empty">Ask about the current episode or a flagged clip.</p>
				<p class="chat-empty-sub">Try: "why is this clip marked?" or "what's unusual about clip 2?"</p>
			</div>
		{:else}
			{#each chatHistory as msg, idx (`${msg.ts}-${idx}`)}
				{#if msg.role === 'user'}
					<div class="chat-row-user">
						<p class="chat-bubble-user">{msg.content}</p>
					</div>
				{:else}
					<div class="chat-row-assistant">
						<div class="chat-assistant-stack">
							<div class="chat-bubble-assistant">
								{#if messageContent(msg.content)}
									<p class="chat-text-assistant">{messageContent(msg.content)}</p>
								{/if}
								{#if msg.citation || clipStatus(msg.status)}
									<div class="chat-meta-row">
										{#if msg.citation}
											<button class="chat-citation" onclick={() => togglePreview(msg.citation!, messagePreviewKey(msg, idx))}>
												episode {msg.citation.episode_id}
												{#if msg.citation.frame_start !== undefined && msg.citation.frame_end !== undefined}
													· frames {msg.citation.frame_start}-{msg.citation.frame_end}
												{:else}
													· frame {msg.citation.frame_index}
												{/if}
											</button>
										{/if}
										{#if clipStatus(msg.status)}
											<span class="chat-status">{clipStatus(msg.status)}</span>
										{/if}
									</div>
								{/if}
								{#if msg.citation}
									{@const key = citationKey(msg.citation)}
									{#if openPreviewKey === messagePreviewKey(msg, idx)}
										<div class="chat-preview-shell">
											{#if previewLoading[key]}
												<p class="chat-preview-empty">loading clip…</p>
											{:else if previewError[key]}
												<p class="chat-preview-empty">{previewError[key]}</p>
											{:else if previewCache[key]}
												<div class="chat-preview-row">
													{#each previewCache[key].byCamera as previewCamera (previewCamera.camera)}
														<div class="chat-preview-card">
															{#if previewFrameUrl(previewCache[key], previewCamera.camera)}
																<img
																	src={previewFrameUrl(previewCache[key], previewCamera.camera) ?? ''}
																	alt=""
																	class="chat-preview-media"
																/>
															{/if}
															<span class="chat-preview-label">{cameraLabel(previewCamera.camera)}</span>
														</div>
													{/each}
												</div>
												<div class="chat-preview-footer">
													<div class="chat-preview-meta">
														{#if clusterLabel(msg.citation)}
															<span class="chat-preview-cluster">{clusterLabel(msg.citation)}</span>
														{/if}
													</div>
													<div class="chat-preview-actions">
														<button
															class="chat-review-btn chat-review-approve"
															disabled={validating[key]}
															onclick={() => handleValidate(msg.citation!, 'approved')}
														>
															approve
														</button>
														<button
															class="chat-review-btn chat-review-reject"
															disabled={validating[key]}
															onclick={() => handleValidate(msg.citation!, 'rejected')}
														>
															reject
														</button>
													</div>
												</div>
											{:else}
												<p class="chat-preview-empty">clip preview unavailable</p>
											{/if}
										</div>
									{/if}
								{/if}
							</div>
						</div>
					</div>
				{/if}
			{/each}
			{#if vlmBusy}
				<div class="chat-row-assistant">
					<div class="chat-assistant-stack">
						<p class="chat-thinking"><span class="dot"></span><span class="dot"></span><span class="dot"></span></p>
					</div>
				</div>
			{/if}
		{/if}
	</div>
</div>

<style>
	.chat-col {
		flex: 1;
		min-width: 0;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
		align-items: center;
		scrollbar-width: thin;
		scrollbar-color: rgba(255, 255, 255, 0.08) transparent;
	}
	.chat-log {
		width: 100%;
		max-width: 680px;
		padding: 24px 0 16px;
		display: flex;
		flex-direction: column;
		gap: 0;
	}
	.chat-empty-wrap {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 8px;
		flex: 1;
		padding: 60px 0;
		text-align: center;
	}
	.chat-empty {
		margin: 0;
		font-size: 15px;
		font-weight: 500;
		color: rgba(255, 255, 255, 0.55);
	}
	.chat-empty-sub {
		margin: 0;
		font-size: 13px;
		color: rgba(255, 255, 255, 0.28);
	}
	.chat-row-user {
		display: flex;
		justify-content: flex-end;
		padding: 4px 0;
	}
	.chat-bubble-user {
		margin: 0;
		max-width: 72%;
		background: #1c1c1c;
		border: 1px solid rgba(255, 255, 255, 0.09);
		border-radius: 18px 18px 4px 18px;
		padding: 10px 15px;
		font-size: 14px;
		line-height: 1.55;
		color: rgba(255, 255, 255, 0.88);
		white-space: pre-wrap;
		word-break: break-word;
	}
	.chat-row-assistant {
		display: flex;
		align-items: flex-start;
		gap: 12px;
		padding: 6px 0;
	}
	.chat-assistant-stack {
		flex: 1;
		display: flex;
		flex-direction: column;
		gap: 6px;
	}
	.chat-bubble-assistant {
		max-width: 72%;
		background: #1c1c1c;
		border: 1px solid rgba(255, 255, 255, 0.09);
		border-radius: 18px 18px 18px 4px;
		padding: 10px 15px;
		display: flex;
		flex-direction: column;
		gap: 10px;
	}
	.chat-text-assistant {
		margin: 0;
		font-size: 14px;
		line-height: 1.55;
		color: rgba(255, 255, 255, 0.9);
		white-space: pre-wrap;
		word-break: break-word;
	}
	.chat-meta-row {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}
	.chat-citation {
		border: none;
		background: rgba(255, 255, 255, 0.04);
		color: rgba(255, 255, 255, 0.62);
		border-radius: 999px;
		padding: 4px 10px;
		display: inline-flex;
		align-items: center;
		font: inherit;
		font-size: 11px;
		cursor: pointer;
	}
	.chat-citation:hover {
		color: rgba(255, 255, 255, 0.82);
		background: rgba(255, 255, 255, 0.07);
	}
	.chat-status {
		font-size: 11px;
		color: rgba(255, 255, 255, 0.34);
		letter-spacing: 0.01em;
	}
	.chat-preview-shell {
		display: flex;
		flex-direction: column;
		gap: 10px;
		align-items: flex-start;
	}
	.chat-preview-row {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
	}
	.chat-preview-card {
		position: relative;
		width: 128px;
		aspect-ratio: 16 / 10;
		border-radius: 12px;
		overflow: hidden;
		background: #050505;
	}
	.chat-preview-media {
		width: 100%;
		height: 100%;
		object-fit: cover;
		display: block;
	}
	.chat-preview-label {
		position: absolute;
		left: 8px;
		bottom: 8px;
		font-size: 10px;
		color: #fff;
		mix-blend-mode: difference;
		pointer-events: none;
		user-select: none;
	}
	.chat-preview-footer {
		display: flex;
		width: 100%;
		align-items: center;
		justify-content: space-between;
		gap: 12px;
	}
	.chat-preview-meta {
		display: flex;
		align-items: center;
		gap: 8px;
	}
	.chat-preview-cluster {
		font-size: 11px;
		color: rgba(255, 255, 255, 0.52);
	}
	.chat-preview-actions {
		display: flex;
		align-items: center;
		gap: 8px;
		margin-left: auto;
	}
	.chat-review-btn {
		border: none;
		border-radius: 10px;
		padding: 7px 12px;
		font: inherit;
		font-size: 11px;
		cursor: pointer;
	}
	.chat-review-btn:disabled {
		opacity: 0.45;
		cursor: not-allowed;
	}
	.chat-review-approve {
		background: rgba(255, 255, 255, 0.95);
		color: rgba(0, 0, 0, 0.84);
	}
	.chat-review-reject {
		background: rgba(0, 0, 0, 0.48);
		color: rgba(255, 255, 255, 0.88);
	}
	.chat-preview-empty {
		margin: 0;
		font-size: 11px;
		color: rgba(255, 255, 255, 0.42);
	}
	.chat-thinking {
		margin: 0;
		padding-top: 8px;
		display: flex;
		gap: 4px;
		align-items: center;
	}
	.dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: rgba(255, 255, 255, 0.4);
		animation: bounce 1.2s ease infinite;
	}
	.dot:nth-child(2) {
		animation-delay: 0.2s;
	}
	.dot:nth-child(3) {
		animation-delay: 0.4s;
	}
	@keyframes bounce {
		0%,
		60%,
		100% {
			transform: translateY(0);
			opacity: 0.4;
		}
		30% {
			transform: translateY(-4px);
			opacity: 1;
		}
	}
</style>
