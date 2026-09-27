# GitHub and Gradescope submission checklist

This repository is your assignment submission. Commit and push your work to
the private repository created for you in the course GitHub organization.

Before submitting, verify all of the following:

- [x] Every `TODO` in `models.py` and `train.py` is complete.
- [x] The GitHub Actions **Student checks** workflow passes on the submitted commit.
- [x] `experiments.json` still contains all required experiment IDs.
- [x] The full experiments were run without the `--max-*-samples` flags.
- [x] The generated `results.csv` is committed and pushed.
- [ ] All seven questions in `analysis.md` are answered.
- [x] The `data/` directory and model checkpoints were not committed.
- [ ] Gradescope is linked to the correct private repository and commit/branch.

Useful commands:

```bash
git status
git add models.py train.py run_experiments.py \
  experiments.json results.csv analysis.md
git commit -m "Complete CIFAR-10 classifier assignment"
git push
```

After pushing, open the repository's **Actions** tab and wait for the checks to
finish. Then submit through Gradescope's GitHub integration. Gradescope grades
the exact repository revision selected there; later pushes do not silently
replace an existing submission.
