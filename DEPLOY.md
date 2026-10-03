# Deploying RegressionLab

## 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: RegressionLab"
git branch -M main
# create an empty repo named "regressionlab" on github.com first, then:
git remote add origin https://github.com/YOUR-USERNAME/regressionlab.git
git push -u origin main
```
Replace `YOUR-USERNAME` in README.md and `YOUR NAME` in LICENSE, then commit and push again.

## 2. Check CI
Open the repo's **Actions** tab; the CI workflow should turn green.

## 3. Live web app (Streamlit Community Cloud, free)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app** -> choose your repo, branch `main`, main file `app.py`.
3. Click **Deploy**. You get a public `*.streamlit.app` URL; add it to the README.

## 4. Docker (optional)
```bash
docker build -t regressionlab .
docker run -p 8501:8501 regressionlab
```
