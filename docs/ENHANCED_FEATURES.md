# Enhanced FakeScope Features Guide

## 🎉 New Features Overview

FakeScope has been significantly enhanced with interactive features to provide a comprehensive fact-checking experience. Here's what's new:

---

## 1. 🤖 LLM Provider Selection

### What It Does
Choose between three powerful AI models directly in the UI, each optimized for different use cases.

### Available Models

#### **OpenAI (GPT-4o-mini)**
- **Best for:** Structured, reliable analysis
- **Strengths:**
  - Consistent quality across all queries
  - Fast responses (1-2 seconds)
  - Excellent for professional fact-checking
  - Well-formatted, clear explanations
- **Cost:** ~$0.01-0.03 per analysis
- **When to use:** Production environments, professional reports, need consistency

#### **Google Gemini (1.5 Flash)**
- **Best for:** High-volume usage and cost savings
- **Strengths:**
  - **FREE tier** with 1,500 requests/day
  - Very fast responses
  - Natural, conversational tone
  - Multimodal capable (future: can process images)
- **Cost:** FREE (up to 1,500/day) or ~$0.005-0.01
- **When to use:** Development, testing, high-volume deployments, budget constraints

#### **Perplexity (Sonar)**
- **Best for:** Current events and recent news
- **Strengths:**
  - Real-time web search included
  - Latest information and context
  - Automatically cites additional sources
  - Great for breaking news verification
- **Cost:** ~$0.01-0.05 per analysis
- **When to use:** Breaking news, current events, need latest context

### How to Use
1. Go to **🔍 Analyze** tab
2. Select your preferred model from the dropdown
3. Click the info icon to see why each model is best
4. Run your analysis

---

## 2. 💬 Chat & Debate Feature

### What It Does
After getting an initial analysis, engage in an interactive conversation with the AI to:
- Challenge the verdict
- Ask for more evidence
- Explore alternative perspectives
- Deep-dive into specific aspects

### Features
- **Context-aware:** AI remembers the original claim and analysis
- **Multi-turn conversation:** Build on previous messages
- **Quick prompts:** Pre-written prompts for common questions
- **Chat history:** See the full conversation thread

### How to Use

#### Basic Chat
1. Run an analysis in the **🔍 Analyze** tab
2. Go to **💬 Chat & Debate** tab
3. Type your question or argument
4. Click "Send"

#### Quick Prompts
Use one of the quick prompt buttons:
- **"Why is this fake/true?"** - Get detailed reasoning
- **"Show me evidence"** - See specific supporting/contradicting evidence
- **"I disagree"** - Challenge the AI's assessment

#### Example Conversation
```
You: "Why do you think this is fake? The sources seem credible."

AI: "While the publisher may be reputable, several red flags appear in the claim itself. First, the timeline doesn't match official records. Second, no other major outlets have corroborated this story despite its significance. Third, the language used contains emotional appeals typical of misinformation..."

You: "Show me evidence"

AI: "Here's the specific evidence contradicting this claim:
1. Official CDC database shows no such event on the claimed date
2. Reuters fact-check (link) debunked a similar claim last month
3. The 'expert' quoted doesn't appear in any academic database..."
```

### Tips for Effective Debates
- **Be specific:** Instead of "I disagree," say "I disagree because source X says Y"
- **Ask for clarification:** "What do you mean by 'typical misinformation patterns'?"
- **Challenge with evidence:** "But this government website says otherwise"
- **Explore nuance:** "Could this be partially true?"

---

## 3. ⚖️ Compare Models Feature

### What It Does
Run the same analysis with two different AI models and compare their responses side-by-side.

### Why It's Useful
- **Consensus checking:** Do multiple AIs agree on the verdict?
- **Depth comparison:** Which provides more detailed analysis?
- **Style preference:** Which explanation is clearer for your audience?
- **Confidence validation:** Do both models have similar confidence levels?

### How to Use
1. Run initial analysis with Model A
2. Go to **⚖️ Compare Models** tab
3. Select Model B from dropdown
4. Click **"🔄 Run Comparison"**
5. View side-by-side results

### What to Look For

#### Agreement
- ✅ **Both say FAKE:** High confidence it's false
- ✅ **Both say TRUE:** High confidence it's true
- ⚠️ **Disagreement:** Ambiguous claim, investigate further

#### Quality Indicators
- **Source citations:** Which model provides more external references?
- **Reasoning depth:** Which explains *why* not just *what*?
- **Hedge language:** Appropriate uncertainty vs. overconfidence
- **Actionable insights:** Which helps you make better decisions?

### Example Comparison

| Aspect | OpenAI | Gemini |
|--------|--------|--------|
| **Verdict** | FAKE (confidence: 85%) | FAKE (confidence: 78%) |
| **Sources cited** | 3 external fact-checks | 5 external fact-checks |
| **Explanation length** | 150 words, structured | 200 words, conversational |
| **Key evidence** | Focuses on timeline inconsistency | Focuses on source credibility |
| **Best for** | Professional report | Public communication |

---

## 4. 📊 Deep Analysis Tab

### What It Does
Provides comprehensive insights beyond the basic credibility score.

### Features

#### 📰 Source Analysis
- **Rating breakdown:** Visual chart showing distribution of TRUE/FALSE/MIXED ratings
- **Source count:** How many external fact-checkers agree/disagree
- **Publisher diversity:** Are sources from multiple independent organizations?
- **Detailed source cards:** Expandable details for each fact-check
  - Rating (True/False/Mixed)
  - Claim text
  - Direct link to fact-check
  - Review date

#### 🎯 Model Confidence Visualization
- **Bar chart:** Visual representation of Fake vs. True probabilities
- **Interpretation guide:** What confidence levels mean
  - 90-100%: Very high confidence
  - 70-89%: High confidence
  - 50-69%: Moderate confidence
  - 30-49%: Low confidence (ambiguous)
  - 0-29%: Opposite verdict likely

#### 📈 Text Statistics
- **Character count:** Text length
- **Word count:** Number of words
- **Sentence count:** Complexity indicator
- **Average word length:** Readability metric
- **Usefulness:** Longer, detailed claims are easier to fact-check than vague ones

#### 🏷️ Key Topics
- **Automatic keyword extraction:** Main subjects discussed
- **Topic clustering:** Related themes
- **Context building:** Understand what the claim is about at a glance

### How to Use
1. Run analysis first
2. Go to **📊 Deep Analysis** tab
3. Scroll through sections:
   - Check source consensus
   - Review model confidence
   - Examine text characteristics
   - Identify key topics

### Interpretation Tips

#### Source Analysis
- **5+ sources, all agree:** Very reliable verdict
- **2-3 sources, split verdict:** Investigate further
- **0-1 sources:** Too new or niche to fact-check reliably

#### Confidence Levels
- **High confidence + multiple sources:** Trust the verdict
- **High confidence + no sources:** Model may be overconfident
- **Low confidence + sources:** Sources more reliable than model
- **Low confidence + no sources:** Inconclusive, be skeptical

#### Text Stats
- **Very short text (<100 words):** Hard to analyze, may need more context
- **Medium text (100-500 words):** Ideal for analysis
- **Long text (500+ words):** May contain mix of true/false claims

---

## 5. 📈 Enhanced Dashboard

### What It Does
Track all your fact-checking analyses over time with persistent history.

### Features
- **Full history:** All past analyses in one place
- **Sortable table:** Filter by date, score, verdict
- **Trend visualization:** Line chart showing credibility scores over time
- **Data export:** Copy or download results (via browser)

### Use Cases
- **Personal tracking:** Monitor claims you've checked
- **Team collaboration:** Share fact-checking results
- **Pattern detection:** Notice if certain topics trend fake/true
- **Reporting:** Generate summaries for stakeholders

---

## 🚀 Getting Started

### First Time Setup

1. **Set up API keys** (choose one):
   ```bash
   # Option 1: OpenAI
   export OPENAI_API_KEY="sk-your-key"
   export FAKESCOPE_LLM_PROVIDER="openai"
   
   # Option 2: Gemini (FREE tier!)
   export GEMINI_API_KEY="your-key"
   export FAKESCOPE_LLM_PROVIDER="gemini"
   
   # Option 3: Perplexity
   export PERPLEXITY_API_KEY="pplx-your-key"
   export FAKESCOPE_LLM_PROVIDER="perplexity"
   
   # Optional: Google Fact Check
   export GOOGLE_FACTCHECK_API_KEY="your-key"
   ```

2. **Run the app:**
   ```bash
   source .venv/bin/activate
   python -m streamlit run src/app.py
   ```

3. **Open browser:** http://localhost:8501

### Recommended Workflow

#### Quick Check
1. Paste claim in **🔍 Analyze** tab
2. Select **Gemini** (free, fast)
3. Review verdict and explanation
4. Done! (30 seconds)

#### Thorough Investigation
1. Run initial analysis with **OpenAI** (structured)
2. Check **📊 Deep Analysis** for source breakdown
3. Go to **⚖️ Compare Models** with **Perplexity** (current events)
4. If uncertain, use **💬 Chat & Debate** to explore
5. Save findings from **📈 Dashboard** (5 minutes)

#### Breaking News
1. Use **Perplexity** for real-time context
2. Check **📊 Deep Analysis** - expect few sources (too new)
3. Use **💬 Chat & Debate** to ask about timeline/context
4. Compare with **Gemini** for cost-effective second opinion
5. Monitor over time as more sources emerge

---

## 💡 Best Practices

### Choosing the Right Model

| Scenario | Recommended Model | Why |
|----------|------------------|-----|
| Professional report | OpenAI | Structured, consistent |
| High volume (100+/day) | Gemini | Free tier, fast |
| Breaking news | Perplexity | Real-time search |
| Budget constraints | Gemini | Free tier |
| Need consensus | Compare 2 models | Validation |
| Teaching/Demo | Gemini | Free, natural language |

### Chat & Debate Tips

**Good prompts:**
- ✅ "What specific evidence contradicts the timeline?"
- ✅ "Can you find any credible sources supporting this?"
- ✅ "Explain why the publisher matters here"

**Avoid:**
- ❌ "You're wrong" (not specific enough)
- ❌ "I think..." (ask for evidence instead)
- ❌ Very long prompts (break into smaller questions)

### Interpreting Results

**High Confidence Scenarios:**
```
Credibility: 92/100
Sources: 5 fact-checks, all "FALSE"
→ Very likely fake, trust the verdict
```

**Low Confidence Scenarios:**
```
Credibility: 48/100
Sources: 1 fact-check, "MIXED"
→ Ambiguous, investigate further
```

**Contradiction Scenarios:**
```
Model: 85% TRUE
Sources: 3 "FALSE" ratings
→ Trust sources over model, investigate why model disagrees
```

---

## 🔧 Advanced Features

### Model Comparison Strategies

**Consensus Detection:**
```python
# All models agree = high confidence
OpenAI: FAKE (85%)
Gemini: FAKE (78%)
Perplexity: FAKE (82%)
→ Consensus: Almost certainly FAKE
```

**Disagreement Analysis:**
```python
# Models disagree = investigate
OpenAI: TRUE (65%)
Gemini: FAKE (58%)
Perplexity: TRUE (72%)
→ Investigate: Check sources, topic novelty, model biases
```

### Chat Conversation Patterns

**Socratic Method:**
1. Start with "Why did you conclude X?"
2. For each reason, ask "What evidence supports this?"
3. Challenge strongest evidence: "Could this be explained differently?"
4. Synthesize: "Given these arguments, what's most likely?"

**Devil's Advocate:**
1. Take opposite position: "I believe this is true because..."
2. Ask AI to respond with counter-evidence
3. Refute AI's counter-evidence
4. See if AI can maintain its position or concedes nuance

---

## 📚 Troubleshooting

### "LLM explanation unavailable"
- Check API key is set: `echo $GEMINI_API_KEY`
- Verify provider matches: `echo $FAKESCOPE_LLM_PROVIDER`
- Try different provider

### Chat not responding
- Ensure analysis was run first
- Check chat history isn't too long (clear if needed)
- Verify API key for selected provider

### Model comparison shows identical results
- Bug: Provider may not have switched
- Workaround: Run analysis again with different provider
- Check terminal logs for errors

### No external sources found
- Claim may be too new (not fact-checked yet)
- Claim may be too specific/niche
- Try rephrasing claim for fact-check query

---

## 🎯 Use Case Examples

### Scenario 1: Political Claim
**Claim:** "Senator X voted against healthcare bill"

**Workflow:**
1. Analyze with **Perplexity** (current events)
2. Check **Deep Analysis** for voting record sources
3. **Compare** with **OpenAI** for structured breakdown
4. **Chat:** "Show me the actual voting record"
5. **Result:** Verified voting behavior with context

### Scenario 2: Health Misinformation
**Claim:** "New study shows miracle cure for disease"

**Workflow:**
1. Analyze with **OpenAI** (medical claims need structure)
2. **Deep Analysis:** Check if any medical fact-checkers found it
3. **Chat:** "Is this study peer-reviewed? What journal?"
4. **Compare** with **Gemini** for cost-effective second opinion
5. **Result:** Identified fake study, found real research

### Scenario 3: Breaking News
**Claim:** "Major event just happened in city X"

**Workflow:**
1. Analyze with **Perplexity** (real-time search)
2. **Deep Analysis:** Expect few sources (too recent)
3. **Chat:** "When did this allegedly happen? Any official statements?"
4. Monitor over next hours/days for emerging sources
5. **Result:** Early warning if fake, confirmation if true

---

## 🆕 What's Next?

Planned enhancements:
- [ ] Multi-language support in chat
- [ ] Export analysis reports to PDF
- [ ] Batch analysis (check multiple claims at once)
- [ ] Custom model fine-tuning
- [ ] API access for integrations
- [ ] Chrome extension for real-time checking

---

## 📞 Support

- **Documentation:** See `GUIDE.md`, `GEMINI_SETUP.md`, `PROMPT_CUSTOMIZATION.md`
- **Issues:** https://github.com/enriest/FakeScope/issues
- **Security:** See `SECURITY_WARNING.md` for API key best practices

---

**Enjoy the enhanced FakeScope!** 🎉
