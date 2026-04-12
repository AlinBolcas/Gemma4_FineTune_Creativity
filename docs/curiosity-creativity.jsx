export default function CuriosityCreativity() {
  const W = 1000, H = 820;
  const lx = 230; // curiosity center x
  const rx = 740; // creativity center x
  const midX = 490; // handoff zone

  return (
    <div style={{ background: '#070910', minHeight: '100vh', display: 'flex', justifyContent: 'center', padding: '24px 8px' }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500&display=swap');`}</style>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W, height: 'auto', fontFamily: "'IBM Plex Mono', monospace" }}>
        <defs>
          <pattern id="dots" width="32" height="32" patternUnits="userSpaceOnUse">
            <circle cx="16" cy="16" r="0.65" fill="#0d1820" />
          </pattern>
          <marker id="ac" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <polygon points="0,0 7,3.5 0,7" fill="#1e6060" />
          </marker>
          <marker id="ag" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <polygon points="0,0 7,3.5 0,7" fill="#305030" />
          </marker>
          <marker id="ah" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <polygon points="0,0 7,3.5 0,7" fill="#3a5a4a" />
          </marker>
          <marker id="ax" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <polygon points="0,0 7,3.5 0,7" fill="#4a7a5a" />
          </marker>
        </defs>

        <rect width={W} height={H} fill="#070910" />
        <rect width={W} height={H} fill="url(#dots)" />

        {/* Panel divider — handoff zone */}
        <line x1="490" y1="58" x2="490" y2={H - 50} stroke="#0d1a18" strokeWidth="1" strokeDasharray="2,8" />

        {/* ── SHARED PRIMITIVE FOOTER ── */}
        <rect x="40" y={H - 58} width="920" height="38" rx="3"
          fill="#090c10" stroke="#0e1820" strokeWidth="1" />
        <text x="500" y={H - 40} textAnchor="middle" fill="#1a2830" fontSize="7.5" letterSpacing="2">
          CURIOSITY  →  frames the question space  →  CREATIVITY uses it as steering context
        </text>
        <text x="500" y={H - 26} textAnchor="middle" fill="#141e25" fontSize="7" letterSpacing="1.5">
          both use structured output  ·  both prune on novelty  ·  shared branch-budget mechanism
        </text>

        {/* ═══════════════════════════════
            LEFT — CURIOSITY  (teal)
            ═══════════════════════════════ */}

        <text x={lx} y="30" textAnchor="middle" fill="#2a6a6a" fontSize="10" letterSpacing="3" fontWeight="500">CURIOSITY</text>
        <text x={lx} y="46" textAnchor="middle" fill="#142828" fontSize="7.5" letterSpacing="1">socratic engine  ·  opens questions, never answers</text>

        {/* ── Stage 0: local novelty signal (input) ── */}
        <rect x="50" y="60" width="360" height="36" rx="3"
          fill="#080f14" stroke="#0e2828" strokeWidth="1" strokeDasharray="3,4" />
        <text x={lx} y="76" textAnchor="middle" fill="#1e4848" fontSize="8" letterSpacing="0.5">INPUT + LOCAL NOVELTY SCORE</text>
        <text x={lx} y="89" textAnchor="middle" fill="#0e2828" fontSize="7" letterSpacing="0.5">embedding distance from known territory</text>

        <line x1={lx} y1="96" x2={lx} y2="122" stroke="#1e4848" strokeWidth="1.5" markerEnd="url(#ac)" />

        {/* ── Stage 1: MAP ── */}
        <rect x="50" y="126" width="360" height="76" rx="3" fill="#081818" stroke="#1a5050" strokeWidth="1.5" />
        <text x={lx} y="146" textAnchor="middle" fill="#40a0a0" fontSize="9.5" fontWeight="500" letterSpacing="0.5">① MAP CURIOSITY DOMAINS</text>
        <text x="68" y="163" fill="#1a4848" fontSize="7.5" letterSpacing="0.3">global_novelty_estimate · branch_budget</text>
        <text x="68" y="177" fill="#1a4848" fontSize="7.5" letterSpacing="0.3">known_context · hidden_assumptions</text>
        <text x="68" y="191" fill="#1a4848" fontSize="7.5" letterSpacing="0.3">curiosity_domains · seed_questions · frontier_notes</text>

        {/* Domain lenses row */}
        {["assumption", "opposite", "expert", "frontier", "cross-domain"].map((lens, i) => (
          <rect key={i} x={52 + i * 73} y="207" width="69" height="16" rx="2"
            fill="#060e10" stroke="#0e2828" strokeWidth="1" />
        ))}
        {["assumption", "opposite", "expert", "frontier", "cross-domain"].map((lens, i) => (
          <text key={i} x={52 + i * 73 + 34} y="219" textAnchor="middle" fill="#1e4040" fontSize="6.5" letterSpacing="0.2">{lens}</text>
        ))}

        <line x1={lx} y1="226" x2={lx} y2="256" stroke="#1e5050" strokeWidth="1.5" markerEnd="url(#ac)" />

        {/* ── Stage 2: BRANCH ── */}
        <rect x="50" y="260" width="360" height="70" rx="3" fill="#081818" stroke="#1a5050" strokeWidth="1.5" />
        <text x={lx} y="280" textAnchor="middle" fill="#40a0a0" fontSize="9.5" fontWeight="500" letterSpacing="0.5">② EXPAND QUESTION BRANCHES</text>
        <text x="68" y="297" fill="#1a4848" fontSize="7.5" letterSpacing="0.3">per domain: direction · questions[2-3] · why_non_obvious</text>
        <text x="68" y="311" fill="#1a4848" fontSize="7.5" letterSpacing="0.3">curiosity_strength  ·  prune redundant/shallow branches</text>

        {/* Branch fan */}
        {[-72, -36, 0, 36, 72].map((dx, i) => (
          <g key={i}>
            <line x1={lx} y1="330" x2={lx + dx} y2="358"
              stroke={Math.abs(dx) > 50 ? "#1a2828" : "#1a4848"} strokeWidth="1.1" />
            <circle cx={lx + dx} cy="362" r="5"
              fill={Math.abs(dx) > 50 ? "#0c1a18" : "#0a1818"}
              stroke={Math.abs(dx) > 50 ? "#1a2a28" : "#1e5050"} strokeWidth="1.1" />
            {Math.abs(dx) > 50 && (
              <g>
                <line x1={lx + dx - 4} y1="357" x2={lx + dx + 4} y2="367" stroke="#2a2020" strokeWidth="1.2" />
                <line x1={lx + dx + 4} y1="357" x2={lx + dx - 4} y2="367" stroke="#2a2020" strokeWidth="1.2" />
              </g>
            )}
          </g>
        ))}
        <text x="52" y="358" fill="#0e2828" fontSize="7">pruned</text>
        <text x="310" y="358" fill="#0e2828" fontSize="7">pruned</text>

        {/* Converge surviving branches */}
        {[-36, 0, 36].map((dx, i) => (
          <line key={i} x1={lx + dx} y1="367" x2={lx} y2="390"
            stroke="#1a4848" strokeWidth="1.1" />
        ))}
        <circle cx={lx} cy="392" r="5" fill="#0a1818" stroke="#1e5050" strokeWidth="1.2" />

        <line x1={lx} y1="397" x2={lx} y2="420" stroke="#1e5050" strokeWidth="1.5" markerEnd="url(#ac)" />

        {/* ── Stage 3: DISTILL ── */}
        <rect x="50" y="424" width="360" height="80" rx="3" fill="#081818" stroke="#1a5050" strokeWidth="1.5" />
        <text x={lx} y="444" textAnchor="middle" fill="#40a0a0" fontSize="9.5" fontWeight="500" letterSpacing="0.5">③ DISTILL QUESTION SET</text>
        <text x="68" y="461" fill="#1a4848" fontSize="7.5" letterSpacing="0.3">best_questions[3-4]  ·  leverage score</text>
        <text x="68" y="475" fill="#1a4848" fontSize="7.5" letterSpacing="0.3">socratic_scaffold  ·  exploration_direction</text>
        <text x="68" y="489" fill="#1a4848" fontSize="7.5" letterSpacing="0.3">favour questions that unlock multiple branches</text>

        <line x1={lx} y1="504" x2={lx} y2="534" stroke="#1e5050" strokeWidth="1.5" markerEnd="url(#ac)" />

        {/* ── OUTPUT ── */}
        <ellipse cx={lx} cy="558" rx="100" ry="30" fill="#081818" stroke="#2a7070" strokeWidth="1.5" />
        <text x={lx} y="553" textAnchor="middle" fill="#50c0b8" fontSize="9.5" fontWeight="500" letterSpacing="0.5">SOCRATIC OUTPUT</text>
        <text x={lx} y="568" textAnchor="middle" fill="#1e5050" fontSize="7.5" letterSpacing="0.3">scaffold + direction + questions</text>

        {/* ─── HANDOFF ARROW ─── */}
        <line x1="330" y1="558" x2="540" y2="558"
          stroke="#2a5a4a" strokeWidth="1.5" strokeDasharray="5,4" markerEnd="url(#ah)" />
        <text x="435" y="548" textAnchor="middle" fill="#1e3830" fontSize="7" letterSpacing="0.5">steering</text>

        {/* ═══════════════════════════════
            RIGHT — CREATIVITY  (green)
            ═══════════════════════════════ */}

        <text x={rx} y="30" textAnchor="middle" fill="#305830" fontSize="10" letterSpacing="3" fontWeight="500">CREATIVITY</text>
        <text x={rx} y="46" textAnchor="middle" fill="#182818" fontSize="7.5" letterSpacing="1">"everything is a remix"  ·  combinatory synthesis</text>

        {/* ── Stage 1: RESEARCH PLAN ── */}
        <rect x="520" y="60" width="440" height="84" rx="3" fill="#0c180c" stroke="#1e3e1e" strokeWidth="1.5" />
        <text x={rx} y="80" textAnchor="middle" fill="#58a858" fontSize="9.5" fontWeight="500" letterSpacing="0.5">① RESEARCH PLAN</text>
        <text x="536" y="97" fill="#254825" fontSize="7.5" letterSpacing="0.3">complexity · branch_budget[2-6]</text>
        <text x="536" y="111" fill="#254825" fontSize="7.5" letterSpacing="0.3">known_patterns · adjacent_domains · creative_tensions</text>
        <text x="536" y="125" fill="#254825" fontSize="7.5" letterSpacing="0.3">research_queries → synthesis  (if web enabled)</text>

        <line x1={rx} y1="144" x2={rx} y2="170" stroke="#305030" strokeWidth="1.5" markerEnd="url(#ag)" />

        {/* ── Stage 2: BRANCH ── */}
        <rect x="520" y="174" width="440" height="60" rx="3" fill="#0c180c" stroke="#1e3e1e" strokeWidth="1.5" />
        <text x={rx} y="194" textAnchor="middle" fill="#58a858" fontSize="9.5" fontWeight="500" letterSpacing="0.5">② BRANCH</text>
        <text x="536" y="211" fill="#254825" fontSize="7.5" letterSpacing="0.3">N directions of attack  ·  frame · domain · constraint · examples</text>
        <text x="536" y="225" fill="#254825" fontSize="7.5" letterSpacing="0.3">each branch must be structurally distinct</text>

        <line x1={rx} y1="234" x2={rx} y2="258" stroke="#305030" strokeWidth="1.5" markerEnd="url(#ag)" />

        {/* ── Stage 2.5: DEVELOP ── */}
        <rect x="520" y="262" width="440" height="58" rx="3" fill="#0c180c" stroke="#1e3e1e" strokeWidth="1.5" />
        <text x={rx} y="282" textAnchor="middle" fill="#58a858" fontSize="9.5" fontWeight="500" letterSpacing="0.5">② .5  DEVELOP BRANCHES</text>
        <text x="536" y="299" fill="#254825" fontSize="7.5" letterSpacing="0.3">chain_steps[2-5] per branch  ·  exhaust each independently</text>
        <text x="536" y="312" fill="#254825" fontSize="7.5" letterSpacing="0.3">branch_outputs  ·  exhausted_when</text>

        <line x1={rx} y1="320" x2={rx} y2="344" stroke="#305030" strokeWidth="1.5" markerEnd="url(#ag)" />

        {/* ── Stage 3: SELECTION ── */}
        <rect x="520" y="348" width="440" height="60" rx="3" fill="#0c180c" stroke="#1e3e1e" strokeWidth="1.5" />
        <text x={rx} y="368" textAnchor="middle" fill="#58a858" fontSize="9.5" fontWeight="500" letterSpacing="0.5">③ SELECTION</text>
        <text x="536" y="385" fill="#254825" fontSize="7.5" letterSpacing="0.3">score: novelty · relevance · combinability  ·  keep 2-3</text>
        <text x="536" y="399" fill="#1e3818" fontSize="7.5" letterSpacing="0.3">prune convergent branches  ·  reward structural distance</text>

        {/* Prune visual */}
        {[540, 590, rx, 890, 940].map((bx, i) => (
          <g key={i}>
            <line x1={rx} y1="408" x2={bx} y2="432"
              stroke={[0, 4].includes(i) ? "#1a2020" : "#1a3828"} strokeWidth="1.1" />
            <circle cx={bx} cy="436" r="5"
              fill={[0, 4].includes(i) ? "#0e1010" : "#0a1408"}
              stroke={[0, 4].includes(i) ? "#2a2020" : "#1e4028"} strokeWidth="1.1" />
            {[0, 4].includes(i) && (
              <g>
                <line x1={bx - 4} y1="431" x2={bx + 4} y2="441" stroke="#2a1e1e" strokeWidth="1.2" />
                <line x1={bx + 4} y1="431" x2={bx - 4} y2="441" stroke="#2a1e1e" strokeWidth="1.2" />
              </g>
            )}
          </g>
        ))}

        {/* Converge surviving 3 */}
        {[590, rx, 890].map((bx, i) => (
          <line key={i} x1={bx} y1="441" x2={rx} y2="464"
            stroke="#1a3828" strokeWidth="1.1" />
        ))}
        <circle cx={rx} cy="466" r="5" fill="#0a1408" stroke="#1e4028" strokeWidth="1.2" />

        <line x1={rx} y1="471" x2={rx} y2="494" stroke="#305030" strokeWidth="1.5" markerEnd="url(#ag)" />

        {/* ── Stage 4: COMBINATION ── */}
        <rect x="520" y="498" width="440" height="60" rx="3" fill="#0c180c" stroke="#1e3e1e" strokeWidth="1.5" />
        <text x={rx} y="518" textAnchor="middle" fill="#58a858" fontSize="9.5" fontWeight="500" letterSpacing="0.5">④ COMBINATORY MIXING</text>
        <text x="536" y="535" fill="#254825" fontSize="7.5" letterSpacing="0.3">cross-pollinate selected branches  ·  from:[B1,B3] → hybrid</text>
        <text x="536" y="549" fill="#1e3818" fontSize="7.5" letterSpacing="0.3">dead_ends logged  ·  strength score  ·  why_novel required</text>

        {/* Cross-connect lines */}
        {[590, rx, 890].map((bx, i) => (
          <line key={i} x1={bx} y1="558" x2={bx === rx ? bx - 80 : bx === 590 ? bx + 80 : bx - 80} y2="590"
            stroke="#1a3020" strokeWidth="1" strokeDasharray="3,3" />
        ))}
        {[640, rx, 840].map((bx, i) => (
          <g key={i}>
            <circle cx={bx} cy="596" r="7" fill="#0a1408" stroke="#254825" strokeWidth="1.3" />
            <line x1={bx} y1="603" x2={bx} y2="624" stroke="#253525" strokeWidth="1.1" markerEnd="url(#ag)" />
          </g>
        ))}
        {[640, rx, 840].map((bx, i) => (
          <line key={i} x1={bx} y1="628" x2={rx} y2="648"
            stroke="#1a3020" strokeWidth="1.1" />
        ))}

        <line x1={rx} y1="648" x2={rx} y2="668" stroke="#305030" strokeWidth="1.5" markerEnd="url(#ag)" />

        {/* ── Stage 5: FINAL SYNTHESIS ── */}
        <ellipse cx={rx} cy="700" rx="130" ry="34" fill="#0c180c" stroke="#2a5a2a" strokeWidth="1.5" />
        <text x={rx} y="695" textAnchor="middle" fill="#68c868" fontSize="9.5" fontWeight="500" letterSpacing="0.5">⑤ FINAL SYNTHESIS</text>
        <text x={rx} y="711" textAnchor="middle" fill="#254825" fontSize="7.5" letterSpacing="0.3">primary_candidates · output · novelty_notes · best_combination</text>

        {/* → OUT */}
        <line x1={rx + 130} y1="700" x2={rx + 165} y2="700"
          stroke="#305030" strokeWidth="1.5" markerEnd="url(#ag)" />
        <rect x={rx + 165} y="688" width="42" height="24" rx="2" fill="#090e08" stroke="#1a2a18" strokeWidth="1" />
        <text x={rx + 186} y="704" textAnchor="middle" fill="#305030" fontSize="8.5" fontWeight="500">OUT</text>

        {/* Novelty uncertainty mark */}
        <text x={rx + 163} y="726" textAnchor="middle" fill="#1a2a18" fontSize="7.5" letterSpacing="0.5">?  (remix, not proof)</text>

      </svg>
    </div>
  );
}
