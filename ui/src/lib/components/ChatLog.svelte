<script lang="ts">
	type ChatTurn = { role: 'user' | 'assistant'; content: string; ts: number };
	let { chatHistory, vlmBusy } = $props<{ chatHistory: ChatTurn[]; vlmBusy: boolean }>();
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
						<div class="chat-avatar">D</div>
						<p class="chat-text-assistant">{msg.content}</p>
					</div>
				{/if}
			{/each}
			{#if vlmBusy}
				<div class="chat-row-assistant">
					<div class="chat-avatar">D</div>
					<p class="chat-thinking"><span class="dot"></span><span class="dot"></span><span class="dot"></span></p>
				</div>
			{/if}
		{/if}
	</div>
</div>

<style>
	.chat-col {
		flex:1;
		min-width:0;
		overflow-y:auto;
		display:flex;
		flex-direction:column;
		align-items:center;
		scrollbar-width:thin;
		scrollbar-color:rgba(255,255,255,0.08) transparent;
	}
	.chat-log {
		width:100%;
		max-width:680px;
		padding:24px 0 16px;
		display:flex;
		flex-direction:column;
		gap:0;
	}
	.chat-empty-wrap {
		display:flex;
		flex-direction:column;
		align-items:center;
		justify-content:center;
		gap:8px;
		flex:1;
		padding:60px 0;
		text-align:center;
	}
	.chat-empty {
		margin:0;
		font-size:15px;
		font-weight:500;
		color:rgba(255,255,255,0.55);
	}
	.chat-empty-sub {
		margin:0;
		font-size:13px;
		color:rgba(255,255,255,0.28);
	}
	.chat-row-user {
		display:flex;
		justify-content:flex-end;
		padding:4px 0;
	}
	.chat-bubble-user {
		margin:0;
		max-width:72%;
		background:#1c1c1c;
		border:1px solid rgba(255,255,255,0.09);
		border-radius:18px 18px 4px 18px;
		padding:10px 15px;
		font-size:14px;
		line-height:1.55;
		color:rgba(255,255,255,0.88);
		white-space:pre-wrap;
		word-break:break-word;
	}
	.chat-row-assistant {
		display:flex;
		align-items:flex-start;
		gap:12px;
		padding:6px 0;
	}
	.chat-avatar {
		flex-shrink:0;
		width:28px;
		height:28px;
		border-radius:50%;
		background:rgba(255,255,255,0.08);
		border:1px solid rgba(255,255,255,0.1);
		display:flex;
		align-items:center;
		justify-content:center;
		font-size:11px;
		font-weight:600;
		color:rgba(255,255,255,0.6);
		margin-top:2px;
	}
	.chat-text-assistant {
		margin:0;
		flex:1;
		font-size:14px;
		line-height:1.65;
		color:rgba(255,255,255,0.94);
		white-space:pre-wrap;
		word-break:break-word;
		padding-top:4px;
	}
	.chat-thinking {
		margin:0;
		padding-top:8px;
		display:flex;
		gap:4px;
		align-items:center;
	}
	.dot {
		width:6px; height:6px;
		border-radius:50%;
		background:rgba(255,255,255,0.4);
		animation:bounce 1.2s ease infinite;
	}
	.dot:nth-child(2) { animation-delay:0.2s; }
	.dot:nth-child(3) { animation-delay:0.4s; }
	@keyframes bounce {
		0%,60%,100% { transform:translateY(0); opacity:0.4; }
		30% { transform:translateY(-4px); opacity:1; }
	}
</style>
