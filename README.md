# Tytonidae Tympanometry (TyTy)



## Admin Steps
Change to where you want the repository to be and run the following command to get started:
```
git clone https://github.com/danielchegwidden/tytonidae-tympanometry.git
```
You can now change into the TyTy repository and have access to all of the files.

To set up a virtual environment, run:
```
python -m venv venv
```
This assumes that your Python command is ```python```, it may be ```python3```. Remember to activate your virtual environment before running code or installing packages:
```
source venv/bin/activate
```
You can keep your virtual environment updated by installing the latest packages from the requirements file:
```
pip install -r requirements.txt
```

Run the following commands once you have updated your packages:
```
pre-commit install
```

The following git commands are the only ones you need to worry about:
```
git branch -va
git status
git checkout -b <my-branch>
git add .
git commit -m "<add commit message here>"
git push origin <my-branch>
git checkout main
git pull origin main
git merge main
```
Please ask for assistance to get into a habit of using these correctly if unsure.
