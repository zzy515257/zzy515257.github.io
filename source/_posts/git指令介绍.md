# git指令介绍

## 1.强制推送

```
git push --force-with-lease
$ git push -f origin main

```

## 2.更新commit

```
git commit --amend -m "refactor: 更改了if语句缩进错误，将model单独打包到model.py文件中，更改了readme中的错误说法"
```

## 3.添加至暂存区

```
git add . 
```

## 4.git删库

```
rm -rf .git
```

## 5.绑定远程仓库，并命名为 origin

```
git remote add origin https://github.com/用户名/仓库名.git
```

## 6.提交

```
git commit -m "first commit"
```

## 7.将当前分支重命名为 main（如果远程默认分支是 master，就改成 master）

```
git branch -M main
```

## 8. Git 子模块（Submodule）或者嵌套的 Git 仓库（Gitlink)

```
为什么会出现这种情况？
通常是因为你在本地把另一个完整的 Git 仓库（即包含了隐藏的 .git 文件夹的目录）直接复制到了这个 demo_submit 目录中，然后执行了 git add . 和 git commit。

此时，Git 不会把那个目录里的文件存进当前仓库，而是只保存了一个指向那个独立仓库的“快捷方式”（记录了一个特定的 Commit ID）。因为真正的文件并没有上传到当前仓库，所以你在 GitHub 上点击它时，要么没反应，要么会跳转到一个 404 页面，因为它无法在当前仓库找到这个目录的内容。
如何验证？
在本地终端里，你可以运行以下命令：
git ls-tree HEAD "0.demo1文本分类"
如果输出的结果是以 160000 commit ... 开头，说明它确实被当成了子模块/嵌套仓库。

如何解决？（将它变成普通文件夹）
如果你想把这个文件夹里的内容当作普通文件上传到当前仓库，需要按以下步骤操作：

第一步：从当前 Git 仓库中移除这个“快捷方式”（不会删除本地文件）
git rm -r --cached "0.demo1文本分类"
第二步：非常重要！删除这个文件夹内部的隐藏 .git 目录
因为它内部有自己的 Git 版本控制，必须删掉它，Git 才会把它当作普通文件夹。

 Git Bash 
 rm -rf "0.demo1文本分类/.git"
 第三步：重新添加并提交
 git add "0.demo1文本分类"
git commit -m "将嵌套仓库改为普通文件夹"
git push
```

## 9.更新文件

```
取消对整个文件的最追踪
git rm (-r) --cached <完整路径>
从新上传
add
```

