<script lang="ts">
	import { onMount } from 'svelte';
	import './layout.css';

	const { children } = $props();

	let inputValue = $state('');
	let textareaEl: HTMLTextAreaElement | undefined;
	let headerView = $state<'main' | 'chat'>('main');

	const hasInput = $derived(inputValue.trim().length > 0);

	function handleInput(e: Event) {
		const el = e.target as HTMLTextAreaElement;
		el.style.height = 'auto';
		el.style.height = Math.min(el.scrollHeight, 120) + 'px';
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			send();
		}
	}

	function send() {
		if (!hasInput) return;
		const message = inputValue.trim();
		if (message.length === 0) return;
		window.dispatchEvent(new CustomEvent('droid:chat-submit', { detail: { message } }));
		inputValue = '';
		if (textareaEl) {
			textareaEl.style.height = 'auto';
		}
	}

	function setHeaderView(view: 'main' | 'chat') {
		headerView = view;
		window.dispatchEvent(new CustomEvent('droid:view-mode', { detail: { view } }));
	}

	onMount(() => {
		window.dispatchEvent(new CustomEvent('droid:view-mode', { detail: { view: headerView } }));
	});
</script>

<div class="app-shell">
	<header class="app-header">
		<span class="brand">droid loop</span>
		<div class="header-segmented" role="tablist" aria-label="Viewport mode">
			<button
				class="header-segment"
				class:active={headerView === 'main'}
				role="tab"
				aria-selected={headerView === 'main'}
				onclick={() => setHeaderView('main')}
			>
				main
			</button>
			<button
				class="header-segment"
				class:active={headerView === 'chat'}
				role="tab"
				aria-selected={headerView === 'chat'}
				onclick={() => setHeaderView('chat')}
			>
				chat
			</button>
		</div>
	</header>

	<main class="app-main" class:chat-mode={headerView === 'chat'}>
		{@render children()}
	</main>

	<footer class="app-footer">
		<div class="input-bar">
			<textarea
				bind:this={textareaEl}
				bind:value={inputValue}
				placeholder="Message droid loop..."
				rows="1"
				oninput={handleInput}
				onkeydown={handleKeydown}
			></textarea>
			<button
				class="send-btn"
				disabled={!hasInput}
				onclick={send}
				aria-label="Send message"
			>
				<svg
					width="15"
					height="15"
					viewBox="0 0 15 15"
					fill="none"
					xmlns="http://www.w3.org/2000/svg"
				>
					<path
						d="M7.5 13V2M2.5 7L7.5 2L12.5 7"
						stroke="currentColor"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"
					/>
				</svg>
			</button>
		</div>
	</footer>

	<a
		class="github-link"
		href="https://github.com/dunkeln/droid_loop"
		target="_blank"
		rel="noreferrer"
		aria-label="Open GitHub repository"
		title="github.com/dunkeln/droid_loop"
	>
		<svg viewBox="0 0 24 24" aria-hidden="true">
			<path
				fill="currentColor"
				d="M12 2C6.477 2 2 6.589 2 12.25c0 4.53 2.865 8.374 6.839 9.73.5.095.683-.223.683-.495 0-.244-.008-.89-.013-1.747-2.782.621-3.37-1.382-3.37-1.382-.455-1.183-1.11-1.498-1.11-1.498-.908-.636.069-.623.069-.623 1.004.072 1.532 1.055 1.532 1.055.892 1.566 2.341 1.114 2.91.852.091-.664.349-1.114.635-1.37-2.221-.26-4.556-1.14-4.556-5.074 0-1.121.39-2.037 1.029-2.754-.103-.26-.446-1.31.098-2.73 0 0 .84-.276 2.75 1.052A9.358 9.358 0 0 1 12 6.88a9.31 9.31 0 0 1 2.504.348c1.909-1.328 2.747-1.052 2.747-1.052.546 1.42.203 2.47.1 2.73.64.717 1.027 1.633 1.027 2.754 0 3.944-2.339 4.811-4.567 5.066.359.32.678.95.678 1.915 0 1.383-.012 2.498-.012 2.837 0 .275.18.595.688.494C19.138 20.62 22 16.778 22 12.25 22 6.589 17.523 2 12 2Z"
			/>
		</svg>
	</a>
</div>
