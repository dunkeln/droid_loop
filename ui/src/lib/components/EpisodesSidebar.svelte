<script lang="ts">
	type EpisodeCard = {
		episode_id: number;
		thumbnail: string | null;
		thumbnailKind: 'image' | 'video';
		incidents: number;
		flagged: number;
		scanning: boolean;
	};

	let { cards, activeEp, loading = false, onSelectEpisode } = $props<{
		cards: EpisodeCard[];
		activeEp: number | null;
		loading?: boolean;
		onSelectEpisode: (episodeId: number) => void;
	}>();
</script>

<aside class="clips-panel">
	<p class="panel-label">episodes</p>
	<div class="episodes-scroll">
			{#if loading}
				<p class="panel-empty">loading episodes…</p>
			{:else if cards.length === 0}
				<p class="panel-empty">no episodes yet</p>
		{:else}
			{#each cards as card (card.episode_id)}
				<div
					class="clip-card"
					class:clip-active={activeEp === card.episode_id}
					role="button"
					tabindex="0"
					onclick={() => onSelectEpisode(card.episode_id)}
					onkeydown={(e) => {
						if (e.key === 'Enter' || e.key === ' ') {
							e.preventDefault();
							onSelectEpisode(card.episode_id);
						}
					}}
				>
						<div class="clip-thumbs">
							{#if card.thumbnail}
								{#if card.thumbnailKind === 'video'}
									<!-- svelte-ignore a11y_media_has_caption -->
									<video
										src={card.thumbnail}
										class="clip-thumb"
										autoplay
										muted
										loop
										playsinline
										preload="metadata"
									></video>
								{:else}
									<img src={card.thumbnail} alt="" class="clip-thumb" onerror={(e)=>{ (e.target as HTMLImageElement).style.opacity='0'; }} />
								{/if}
							{/if}
							<span class="episode-overlay episode-title">episode {card.episode_id}</span>
						{#if card.flagged > 0}
							<span class="episode-overlay episode-meta">{card.incidents} incidents · {card.flagged} flagged</span>
						{:else if card.scanning}
							<span class="episode-overlay episode-meta">scanning…</span>
						{:else}
							<span class="episode-overlay episode-meta">0 incidents · 0 flagged</span>
						{/if}
					</div>
				</div>
			{/each}
		{/if}
	</div>
</aside>

<style>
	.clips-panel {
		width:250px; flex-shrink:0;
		overflow:hidden; display:flex; flex-direction:column; gap:8px;
		border-left:none;
		padding-left:14px; padding-right:6px;
		background:inherit;
		scrollbar-width:thin; scrollbar-color:rgba(255,255,255,0.08) transparent;
	}
	.panel-label {
		font-size:10px; font-weight:600; letter-spacing:0.1em; text-transform:uppercase;
		color:rgba(255,255,255,0.22); margin:0 0 2px; flex-shrink:0;
		padding:0 0 4px;
		background:transparent;
	}
	.episodes-scroll {
		flex:1; min-height:0;
		overflow-y:auto;
		display:flex; flex-direction:column; gap:8px;
		padding-right:2px;
		background:inherit;
		scrollbar-width:thin; scrollbar-color:rgba(255,255,255,0.08) transparent;
	}
	.panel-empty { font-size:12px; color:rgba(255,255,255,0.18); }
	.clip-card {
		width:100%; max-width:180px; align-self:center;
		flex:0 0 auto;
		border-radius:14px; border:none;
		background:inherit; cursor:pointer; overflow:hidden;
		transition:background-color 0.15s;
	}
	.clip-card:hover { background-color:rgba(255,255,255,0.015); }
	.clip-card.clip-active { background-color:rgba(255,255,255,0.02); }
	.clip-thumbs {
		width:100%; aspect-ratio:1/1; display:grid;
		grid-template-columns:1fr; gap:0;
		background:inherit;
		border-radius:12px; overflow:hidden;
		position:relative;
	}
	.clip-thumb { width:100%; height:100%; object-fit:cover; object-position:center; }
	.episode-overlay {
		position:absolute;
		color:#fff;
		mix-blend-mode:difference;
		pointer-events:none;
		user-select:none;
	}
	.episode-title {
		left:8px;
		top:8px;
		font-size:11px;
		font-weight:500;
		letter-spacing:0.04em;
		text-transform:uppercase;
	}
	.episode-meta {
		left:8px;
		bottom:8px;
		font-size:10px;
		font-weight:500;
		letter-spacing:0.02em;
		text-transform:lowercase;
	}
</style>
