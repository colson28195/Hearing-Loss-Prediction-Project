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
On windows this becomes:
```
.\venv\Scripts\activate
```
You can keep your virtual environment updated by installing the latest packages from the requirements file:
```
pip install -r requirements.txt
```

Run the following commands once, once you have updated your packages:
```
pre-commit install
```

### Git Process
Get the latest copy of the repo from GitHub
```
git fetch
```
Update the ```main``` branch on your local machine
```
git checkout main
git pull origin main
```
Update your local development branch with the latest work
```
git checkout <my-branch>
git merge main
```
The steps above here are good to run on a regular basis/before making changes to make sure you are working on the latest version of the code.

On your local development branch, make changes and write code, then state the changes, you can use ```git status``` to check where your changes are in the Git process.
```
git add .
```
Commit the staged changed
```
git commit -m "<my-commit-message>
```
Push the changes from your local machine to GitHub
```
git push origin <my-branch>
```
Go to GitHub and raise a Pull Request from your development branch to ```main```. Once this has been reviewed and approved by someone else, then start from the top again.
