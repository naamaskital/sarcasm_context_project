# Interesting Error Analysis Examples

These examples were automatically extracted from the random-context ablation predictions.

## Counts

- TF-IDF context helped: 152
- TF-IDF random context hurt: 372
- Sentence Transformer joint context helped: 419
- Sentence Transformer separate true/random same prediction: 1661
- All main models wrong: 306


# TF-IDF: context helped

## Example 1

**True label:** not_sarcastic

**Context:** It's predicated on the fact that Smart or Bradley get packaged for say, Blake Griffin. So you would be trading for Blake and then signing waiters to fill a hole in the bench.

**Random context:** America is exporting freedom.

**Comment:** Ahhh yeah I misread the post, my mistake!

- `tfidf_comment_only_pred`: sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---

## Example 2

**True label:** sarcastic

**Context:** Has he ever claimed he could? It's only Fox News that seems to blame Obama when gas prices are high and that his polices as the reason for the high prices and then when the prices go down they say he has nothing to do with it and that in fact the low prices are bad for the economy.

**Random context:** She's definitely cute. But its amazing how quickly my brain turned off when I saw the last pic. It's like she was no longer a woman to me, those hairy legs negated everything

**Comment:** And thankfully no left leaning news media ever blamed bush when prices were high.

- `tfidf_comment_only_pred`: not_sarcastic
- `tfidf_true_context_plus_comment_pred`: sarcastic
- `tfidf_random_context_plus_comment_pred`: not_sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---

## Example 3

**True label:** not_sarcastic

**Context:** Just don't be surprised if you get caught in a sexual situation that you can't avoid with the manager of this team... because of the implication

**Random context:** Yeah, because that's exactly the end goal of terrorism: eliminating God from beer ads.

**Comment:** He invited me to his houseboat this weekend to go for a trip out to int'l waters, but I'm sure it's safe

- `tfidf_comment_only_pred`: sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: not_sarcastic
- `st_true_context_plus_comment_separate_pred`: not_sarcastic
- `st_random_context_plus_comment_separate_pred`: not_sarcastic

---

## Example 4

**True label:** sarcastic

**Context:** Best Anime of Summer 2014

**Random context:** i'm the only one that doesn't watch runefest stream?

**Comment:** Didn't expect this

- `tfidf_comment_only_pred`: not_sarcastic
- `tfidf_true_context_plus_comment_pred`: sarcastic
- `tfidf_random_context_plus_comment_pred`: not_sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: not_sarcastic
- `st_true_context_plus_comment_separate_pred`: sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---

## Example 5

**True label:** sarcastic

**Context:** And got raped.

**Random context:** I agree with you, but I think that it's important to realize that there is an opportunity cost involved in doing so. If you make a course like calculus mandatory in a 4 year nursing program it comes at a cost -- that's one less course directly related to the field of nursing that students do not have time to rake. I do appreciate the ancillary benefits to taking a course like calculus, in that it gives you a diverse set of problem solving skills,...

**Comment:** Better to be raped than to have my infant daughter kidnapped, murdered and have her blood drained to make matzo balls, right?

- `tfidf_comment_only_pred`: not_sarcastic
- `tfidf_true_context_plus_comment_pred`: sarcastic
- `tfidf_random_context_plus_comment_pred`: not_sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---


# TF-IDF: random context hurt

## Example 1

**True label:** not_sarcastic

**Context:** It's predicated on the fact that Smart or Bradley get packaged for say, Blake Griffin. So you would be trading for Blake and then signing waiters to fill a hole in the bench.

**Random context:** America is exporting freedom.

**Comment:** Ahhh yeah I misread the post, my mistake!

- `tfidf_comment_only_pred`: sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---

## Example 2

**True label:** not_sarcastic

**Context:** My 21st Birthday Splurge...

**Random context:** I've never really understood this. Racism is about superiority, if you just generalize a race isn't that just generalizing, with racism being a negative generalization?

**Comment:** $77.94, please pull up to the first window.

- `tfidf_comment_only_pred`: not_sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: not_sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: not_sarcastic
- `st_random_context_plus_comment_separate_pred`: not_sarcastic

---

## Example 3

**True label:** sarcastic

**Context:** Has he ever claimed he could? It's only Fox News that seems to blame Obama when gas prices are high and that his polices as the reason for the high prices and then when the prices go down they say he has nothing to do with it and that in fact the low prices are bad for the economy.

**Random context:** She's definitely cute. But its amazing how quickly my brain turned off when I saw the last pic. It's like she was no longer a woman to me, those hairy legs negated everything

**Comment:** And thankfully no left leaning news media ever blamed bush when prices were high.

- `tfidf_comment_only_pred`: not_sarcastic
- `tfidf_true_context_plus_comment_pred`: sarcastic
- `tfidf_random_context_plus_comment_pred`: not_sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---

## Example 4

**True label:** not_sarcastic

**Context:** Whew I'm out of cat glasses gifs.

**Random context:** I'm sure this is a repost, but it's always worth another watch...George Carlin --Religion is Bullshit

**Comment:** This is one of those *man if I wasn't broke I'd gild the fuck out of you* moments.

- `tfidf_comment_only_pred`: not_sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: not_sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: not_sarcastic
- `st_random_context_plus_comment_separate_pred`: not_sarcastic

---

## Example 5

**True label:** sarcastic

**Context:** SOD-Kill yourself

**Random context:** Nobody made that claim or anything even remotely close though

**Comment:** This isn't System of a Down!

- `tfidf_comment_only_pred`: sarcastic
- `tfidf_true_context_plus_comment_pred`: sarcastic
- `tfidf_random_context_plus_comment_pred`: not_sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---


# Sentence Transformer joint: true context helped over random context

## Example 1

**True label:** not_sarcastic

**Context:** My 21st Birthday Splurge...

**Random context:** I've never really understood this. Racism is about superiority, if you just generalize a race isn't that just generalizing, with racism being a negative generalization?

**Comment:** $77.94, please pull up to the first window.

- `tfidf_comment_only_pred`: not_sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: not_sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: not_sarcastic
- `st_random_context_plus_comment_separate_pred`: not_sarcastic

---

## Example 2

**True label:** not_sarcastic

**Context:** Whew I'm out of cat glasses gifs.

**Random context:** I'm sure this is a repost, but it's always worth another watch...George Carlin --Religion is Bullshit

**Comment:** This is one of those *man if I wasn't broke I'd gild the fuck out of you* moments.

- `tfidf_comment_only_pred`: not_sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: not_sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: not_sarcastic
- `st_random_context_plus_comment_separate_pred`: not_sarcastic

---

## Example 3

**True label:** sarcastic

**Context:** Aim assist for PC? What the actual fuck?

**Random context:** How do I send a package (a book) from my house to another location in Toronto. I have never done it before so I need help.

**Comment:** This is not even far enough, I want the game to just play it's self too.

- `tfidf_comment_only_pred`: not_sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: not_sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: not_sarcastic
- `st_true_context_plus_comment_separate_pred`: not_sarcastic
- `st_random_context_plus_comment_separate_pred`: not_sarcastic

---

## Example 4

**True label:** sarcastic

**Context:** Witcher 3 comes DRM free, with bonus physical content: 'Not only making the asking price easier to stomach but also making you feel valued as a customer.'

**Random context:** My god oj needs to cut that shit off his head

**Comment:** Thanks, didn't know that,

- `tfidf_comment_only_pred`: sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: not_sarcastic
- `st_true_context_plus_comment_separate_pred`: sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---

## Example 5

**True label:** sarcastic

**Context:** Best Anime of Summer 2014

**Random context:** i'm the only one that doesn't watch runefest stream?

**Comment:** Didn't expect this

- `tfidf_comment_only_pred`: not_sarcastic
- `tfidf_true_context_plus_comment_pred`: sarcastic
- `tfidf_random_context_plus_comment_pred`: not_sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: not_sarcastic
- `st_true_context_plus_comment_separate_pred`: sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---


# Sentence Transformer separate: true and random context gave same prediction

## Example 1

**True label:** not_sarcastic

**Context:** It's predicated on the fact that Smart or Bradley get packaged for say, Blake Griffin. So you would be trading for Blake and then signing waiters to fill a hole in the bench.

**Random context:** America is exporting freedom.

**Comment:** Ahhh yeah I misread the post, my mistake!

- `tfidf_comment_only_pred`: sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---

## Example 2

**True label:** sarcastic

**Context:** Massive gunfight at condo during bikefest

**Random context:** fucking idiot... the constitution has no place no politics

**Comment:** They're ruining our city!

- `tfidf_comment_only_pred`: sarcastic
- `tfidf_true_context_plus_comment_pred`: sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---

## Example 3

**True label:** not_sarcastic

**Context:** Isn't a thing kind of pornographic if you jerk to it?

**Random context:** Monster energy drinks should be a controlled substance, I'm tired of seeing ten year olds walking around with them.

**Comment:** You clearly haven't jerked it to Thomas the tank engine before.

- `tfidf_comment_only_pred`: sarcastic
- `tfidf_true_context_plus_comment_pred`: sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: not_sarcastic
- `st_random_context_plus_comment_joint_pred`: not_sarcastic
- `st_true_context_plus_comment_separate_pred`: not_sarcastic
- `st_random_context_plus_comment_separate_pred`: not_sarcastic

---

## Example 4

**True label:** sarcastic

**Context:** It's because only megadouches need to restock their monster supplies at a gas station. The rest of us buy them at grocery stores with the rest of our shopping.

**Random context:** In an undertale chat on steam, a month or two ago, we were talking about how sjws would latch onto this for sure, as tumblr was full of fan art. Really sad to see a great game stained by sjws

**Comment:** YEAH BUDDY TELL YOURSELF THAT IN YOUR 1987 HATCHBACK HONDA CIVIC WITH THE SPOILER

- `tfidf_comment_only_pred`: sarcastic
- `tfidf_true_context_plus_comment_pred`: sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: not_sarcastic
- `st_random_context_plus_comment_joint_pred`: not_sarcastic
- `st_true_context_plus_comment_separate_pred`: not_sarcastic
- `st_random_context_plus_comment_separate_pred`: not_sarcastic

---

## Example 5

**True label:** not_sarcastic

**Context:** My 21st Birthday Splurge...

**Random context:** I've never really understood this. Racism is about superiority, if you just generalize a race isn't that just generalizing, with racism being a negative generalization?

**Comment:** $77.94, please pull up to the first window.

- `tfidf_comment_only_pred`: not_sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: not_sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: not_sarcastic
- `st_random_context_plus_comment_separate_pred`: not_sarcastic

---


# All main true-context models were wrong

## Example 1

**True label:** not_sarcastic

**Context:** Yeah seriously only 57%? Should be more like 97% (because you know 3% morons are likely). Teaching kids that you get something anyway even for losing is a TERRIBLE life lesson, is that how sales jobs work? Is that how most jobs work? HELL NO!

**Random context:** Precisely... That or the leaf meltdown of 2013

**Comment:** well a sales job guy WOULD love everyone to get an award

- `tfidf_comment_only_pred`: sarcastic
- `tfidf_true_context_plus_comment_pred`: sarcastic
- `tfidf_random_context_plus_comment_pred`: sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: not_sarcastic
- `st_true_context_plus_comment_separate_pred`: sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---

## Example 2

**True label:** sarcastic

**Context:** This sounds like a terrible idea

**Random context:** Why is everybody ignoring Scott? Hi guys. I am getting infuriated every time I get on the sub, and it is because of **the box**. Scott said that inside the box is the pieces of the story put together, so why is everybody ignoring him? Have very few people read the announcement, are they stupid like the 8bitgaming youtube channel, or are they just ignoring him? I'm genuinely confused.

**Comment:** We'll meet at the bottom of the stairs by the blue bridge.

- `tfidf_comment_only_pred`: not_sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: not_sarcastic
- `st_true_context_plus_comment_joint_pred`: not_sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: not_sarcastic
- `st_random_context_plus_comment_separate_pred`: not_sarcastic

---

## Example 3

**True label:** sarcastic

**Context:** Best material for a wick? I searched here and didn't find a clear answer. What's the best material? Is it organic cotton, silica, medical gauze, etc? I want the best flavor, but don't know what to use. Thanks

**Random context:** The whole quote is deep so it's not actually stupid

**Comment:** Quartz man!

- `tfidf_comment_only_pred`: sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: not_sarcastic
- `st_true_context_plus_comment_joint_pred`: not_sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: not_sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---

## Example 4

**True label:** not_sarcastic

**Context:** We can't get on a plane without prior immigration visas. They will check all the documents before allowing us to board the plane. The only way is to smuggle ourselves into somewhere in Europe.

**Random context:** Maggie being dumped since Nestle pulled it from Indian market on accusation of excesses lead content been found in it.

**Comment:** Why can't you arrive as tourists and then apply for refugee status?

- `tfidf_comment_only_pred`: sarcastic
- `tfidf_true_context_plus_comment_pred`: sarcastic
- `tfidf_random_context_plus_comment_pred`: not_sarcastic
- `st_true_context_plus_comment_joint_pred`: sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: sarcastic
- `st_random_context_plus_comment_separate_pred`: sarcastic

---

## Example 5

**True label:** sarcastic

**Context:** Suggestion: Make the left click option on "unf" potions "mix", along with vials of water Just a QOL change that i think would make herblore a lot more handy!

**Random context:** That's the thing about freedom, you have to not force your ways on other people even if you think they are making mistakes.

**Comment:** Nty devalues my portable wells.

- `tfidf_comment_only_pred`: not_sarcastic
- `tfidf_true_context_plus_comment_pred`: not_sarcastic
- `tfidf_random_context_plus_comment_pred`: not_sarcastic
- `st_true_context_plus_comment_joint_pred`: not_sarcastic
- `st_random_context_plus_comment_joint_pred`: sarcastic
- `st_true_context_plus_comment_separate_pred`: not_sarcastic
- `st_random_context_plus_comment_separate_pred`: not_sarcastic

---

