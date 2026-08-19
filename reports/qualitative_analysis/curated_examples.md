# Curated Qualitative Examples

These examples were selected from the deterministic categories produced by `scripts/23_qualitative_error_analysis.py`. They are not hand-picked from the full test set without constraints: the script first defines outcome categories, then samples within those categories. The examples below were chosen from that sampled output because they are especially interpretable for the report.

## 1. MiniLM: context helps reveal sarcasm

**Context**  
> I understand when people say the project will fail but calling it a scam has always been the most retarded thing I could imagine someone saying. Scammers don't collect millions... and then put it right back into the scam. They wouldn't have taken all this time building offices, hiring people (especially fucking Mark Hamill), buying equipment, getting into different gaming events, etc if they didn't at least intend on actually making this happen. Like I said there is always a chance the game will fail or just won't live up to the hype but a scam? Hell no...

**Reply**  
> Someone tell those 50+ actors they're working on a scam lol..

**Gold label:** sarcastic  
**MiniLM comment-only:** non-sarcastic  
**MiniLM dual embeddings:** sarcastic

**Interpretation.** The reply alone is short and could plausibly be read literally. The parent message establishes that the speaker rejects the “scam” claim; against that context, the reply becomes an ironic restatement. This is a clean case where conversational information resolves pragmatic ambiguity rather than merely contributing more topic words.

---

## 2. MiniLM: context helps avoid a false sarcasm prediction

**Context**  
> Every time I come up to Memphis it's a good time and thank you tiger fans for the great fun

**Reply**  
> this right here is why I wanted Memphis in the Big XII

**Gold label:** non-sarcastic  
**MiniLM comment-only:** sarcastic  
**MiniLM dual embeddings:** non-sarcastic

**Interpretation.** The reply contains an evaluative statement that can be difficult to interpret without knowing the conversational stance. The positive parent message makes the reply's positive intent coherent and helps the dual representation avoid a false positive. This shows that context can help not only by detecting hidden sarcasm but also by confirming sincerity.

---

## 3. Representation matters: TF-IDF is hurt while MiniLM is helped

**Context**  
> Freezing/Microstuttering 2-4 times for 1-2 seconds during preperation phase. 3-4 Months old issue still not fixxed. My specs: Gtx970 G1 i5 760 quad core 3.5ghz 8gb RAM DDR3 1664MHz Win 7 Game Running on a Samsung SSD , recording with Shadowplay on an HDD. Only game that has performance issues on my rig. During the rest of the game my fps is on average 70-100 fps. Ultra settings, HBAO disabled, Post PRocess AA OFF, AA at TAA, Textures on Very High, Lens & Bloom effects disabled, Shaders on Medium.

**Reply**  
> Download more RAM

**Gold label:** sarcastic  
**TF-IDF comment-only:** sarcastic  
**TF-IDF context+comment:** non-sarcastic  
**MiniLM comment-only:** non-sarcastic  
**MiniLM dual embeddings:** sarcastic

**Interpretation.** This is the clearest illustration of the project's central representation finding. The long technical context adds many high-frequency topical terms; naive TF-IDF concatenation can dilute the compact sarcastic cue in the reply. In contrast, MiniLM keeps the two messages as separate semantic representations and can use the mismatch between a serious troubleshooting description and the absurd advice “Download more RAM.” The same context therefore hurts one integration strategy and helps another.

---

## 4. TF-IDF: useful reply cues are diluted by added context

**Context**  
> Cheap DIY Leap Motion mount for the Vive.

**Reply**  
> I don't know man, rubber bands are pretty expensive

**Gold label:** sarcastic  
**TF-IDF comment-only:** sarcastic  
**TF-IDF context+comment:** non-sarcastic

**Interpretation.** The sarcastic signal is already strong in the reply: calling rubber bands “pretty expensive” contradicts ordinary expectations. Adding the parent text is not necessary for the lexical classifier and instead changes the feature mixture enough to flip a correct prediction. This helps explain why the full-corpus TF-IDF score decreases when context is concatenated naively.

---

## 5. Same-subreddit hard negative breaks a context-dependent prediction

**True context**  
> Opinion: Make all motorway signs bilingual and drop the English verse of the national anthem

**Reply**  
> I vote Klingon as the 2nd language!

**Gold label:** sarcastic  
**Prediction with true context:** sarcastic  
**Prediction with same-subreddit wrong context:** non-sarcastic

**Wrong context**  
> Questions get even harder for milk giant

**Interpretation.** With the true parent message, “Klingon as the 2nd language” is an exaggerated response to a language-policy discussion and the sarcastic relation is easy to reconstruct. A wrong context from the same community/topic environment removes that relation and flips the decision. This supports the claim that at least some examples genuinely depend on the conversational pairing rather than on the reply alone.

---

## 6. Semantic hard negative exposes the limit of embedding-based context use

**True context**  
> A hilarious letter from an entitled abusive ex.

**Reply**  
> Ahh such a sweetheart

**Gold label:** sarcastic  
**Prediction with true context:** sarcastic  
**Prediction with semantically similar wrong context:** non-sarcastic

**Semantically similar wrong context**  
> Friend has a "coexist" sticker on her car. She got this letter because of it.

**Cosine similarity:** 0.4192

**Interpretation.** The sarcastic reading depends on the polarity contrast between an abusive ex and “such a sweetheart.” The semantically related but incorrect context still describes a letter and interpersonal conflict, yet it does not reproduce the exact contradiction. The resulting prediction flip illustrates why semantic similarity alone is not equivalent to the true conversational relation.

---

## Complementary population-level evidence

The selected examples are illustrative rather than the main evidence. The full category counts show the same trends at scale:

- TF-IDF context helped: 9,652 examples
- TF-IDF context hurt: 11,646 examples
- MiniLM context helped: 4,361 examples
- MiniLM context hurt: 3,899 examples
- TF-IDF hurt while MiniLM helped on the same example: 417 examples
- true-context correct -> random-context wrong: 378 examples in the focused hard-context diagnostic
- true-context correct -> same-subreddit wrong: 360 examples
- true-context correct -> semantic-similar wrong: 264 examples
- semantically similar wrong context left the prediction unchanged: 2,352 examples

Together, the aggregate metrics and these cases support a cautious conclusion: context can contain useful pragmatic information, but its benefit depends strongly on how the model represents and integrates the conversational pair.
