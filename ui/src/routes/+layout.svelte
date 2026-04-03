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
</div>
