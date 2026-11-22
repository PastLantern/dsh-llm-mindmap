# DeepSeek Harness 伴学导图

伴学导图是面向 DeepSeek Harness 的社区学习插件。你可以把知识点整理成导图，为每个节点写按节编号的笔记，选中原句发起追问，再把值得保留的回答收进笔记。复习卡按到期日排列，学习数据保存在本地，并同步导出 Markdown。

**当前版本 `0.1.1`** · [macOS 安装包](./dsh-llm-mindmap-0.1.1.tgz?raw=true) · [Windows 安装包](./dsh-llm-mindmap-0.1.1-win.1.tgz?raw=true) · [GitHub Releases](https://github.com/PastLantern/dsh-llm-mindmap/releases/tag/v0.1.1) · [大模型知识库](./nodes/llm-root.md)

## 在 macOS 上安装到 Harness

需要 Node.js 22 或更高版本，以及 DeepSeek Harness 命令行工具 `dsh`。把安装包下载到 `Downloads` 后，在终端执行下面两条命令。

```sh
dsh plugin --profile web add "$HOME/Downloads/dsh-llm-mindmap-0.1.1.tgz"
dsh --profile web --no-open
```

也可以在 Harness 的「插件 → 添加插件」里粘贴安装包的绝对路径。安装完成后启用插件，按界面提示重启当前 profile。

## 在 Windows 上安装到 Harness

确认 DeepSeek Harness 的 Web profile 可以启动。将 Windows 安装包下载到 `Downloads` 后，在 PowerShell 执行。

```powershell
$bundle = Join-Path $HOME 'Downloads/dsh-llm-mindmap-0.1.1-win.1.tgz'
dsh.cmd plugin --profile web add ($bundle -replace '\\', '/')
dsh.cmd --profile web --no-open
```

如果安装的命令是 `dsh.exe`，将上面的 `dsh.cmd` 换成 `dsh.exe`。也可以在 Harness 的「插件 → 添加插件」里粘贴 Windows 安装包的绝对路径。

## 打开伴学导图

打开 Harness 后，从会话顶栏的导图图标或右侧栏「+ → 伴学导图」进入。全新工作区会载入「大模型知识库」和「算法面试刷题库」两个示例。

## 在会话里学习

1. **读笔记**。点击导图节点，查看核心框架和后续小节。用小节引用串联内容，移动小节后显示编号会随之更新。
2. **对原句追问**。选中一句话，点击「追问这句」，输入第一个问题。Harness 会打开一个以该句及所在小节为上下文的独立会话。可以用「回到原会话」继续原来的讨论。
3. **把回答收进导图**。点击回答下方的「存到导图」，编辑草稿或提交待审。Agent 提出的导图改动可以在「待审」里调整后再收录。
4. **做复习卡**。从「复习」打开到期卡片，先回忆再看答案。给自己打 `1`、`2` 或 `3` 分，系统会据此安排下次复习。

## 独立预览

只想试用导图、笔记和复习卡时，可以解压安装包，用 Node.js 启动本地预览。模型追问在 Harness 中使用。

### macOS

```sh
mkdir -p "$HOME/Downloads/dsh-llm-mindmap"
tar -xzf "$HOME/Downloads/dsh-llm-mindmap-0.1.1.tgz" -C "$HOME/Downloads/dsh-llm-mindmap"
cd "$HOME/Downloads/dsh-llm-mindmap/package"
npm install
npm start
```

### Windows

```powershell
$bundle = Join-Path $HOME 'Downloads/dsh-llm-mindmap-0.1.1-win.1.tgz'
$preview = New-Item -ItemType Directory -Path '.\mindmap-preview' -ErrorAction Stop
tar.exe -xzf $bundle -C $preview.FullName
if ($LASTEXITCODE -ne 0) { throw '插件包解压失败' }
Set-Location -LiteralPath (Join-Path $preview.FullName 'package')
npm.cmd install --omit=dev
npm.cmd start
```

打开 [本地预览](http://127.0.0.1:3090/llm-mindmap)。想运行包内自动化测试，在同一目录执行 `npm test`，Windows 上可以执行 `npm.cmd test`。

Windows 版已验证全新安装、独立预览、`Ctrl+K` 搜索、`Ctrl+Enter` 保存和 Markdown 导出。在 Harness Web 中还验证了侧栏、划词独立追问和返回主会话。追问测试使用本地模型替身。

## 大模型知识库

仓库里有 15 篇节点笔记和 11 张题卡，可以直接在 GitHub 阅读。从[总览](./nodes/llm-root.md)进入，或按主题选择。

| 主题 | 节点笔记 | 配套题卡 |
| --- | --- | --- |
| [模型架构](./nodes/cat-arch.md) | [GQA](./nodes/node-gqa.md)、[RoPE](./nodes/node-rope.md)、[RMSNorm 与 SwiGLU](./nodes/node-norm.md) | [实现 GQA](./problems/prob-gqa-code.md)、[推导 KV 开销](./problems/prob-kv-comp.md)、[推导 RoPE](./problems/prob-rope-math.md)、[计算 SwiGLU 参数量](./problems/prob-norm-swiglu.md) |
| [后训练](./nodes/cat-post.md) | [SFT 与 LoRA](./nodes/node-sft.md)、[PPO、DPO 与 GRPO](./nodes/node-grpo.md) | [SFT 与 LoRA 排查](./problems/prob-sft-lora.md)、[比较对齐算法](./problems/prob-rl-grpo.md) |
| [推理系统](./nodes/cat-sys.md) | [KV Cache 与 vLLM](./nodes/node-vllm.md) | [估算显存与吞吐](./problems/prob-kv-roofline.md)、[判断前缀缓存命中](./problems/prob-prefix-caching.md) |
| [Agent 与 Harness](./nodes/cat-agent.md) | [递归自我改进](./nodes/node-rsi.md)、[闭环与护栏](./nodes/node-rsi-notes.md)、[论文导读](./nodes/node-rsi-papers.md)、[长会话缓存与隔离](./nodes/node-harness.md) | [长会话缓存设计](./problems/prob-harness-cache.md)、[设计自我改进评估](./problems/prob-rsi-guard.md)、[推导 STaR](./problems/prob-rsi-star.md) |

## 本地数据

插件把数据写到工作区的 `.llm-mindmap/`。`vaults.json` 是知识库列表，`vaults/<id>/mindmap.v2.json` 保存该知识库的导图、笔记和复习状态。每个知识库的 `markdown/` 目录会同步生成 `INDEX.md`、`nodes/` 和 `problems/`，可以在 Obsidian 中打开。

## 许可

仓库的知识库文字采用 [CC BY 4.0](./LICENSE)。插件安装包在包内 `package.json` 声明 MIT，随包分发的第三方组件附有各自的许可。
