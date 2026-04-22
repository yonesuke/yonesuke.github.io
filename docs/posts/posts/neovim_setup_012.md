---
title: "Neovim 0.12 入門"
date: 2026-04-22
slug: neovim-setup-012
draft: false
authors:
  - yonesuke
categories:
    - Neovim
    - Text Editor
---

コーディングアージェントを利用しているとターミナルから抜け出すのが億劫になってきました。
もともとVSCodeを使っていたのですが、Neovimも気になっていたので、Neovimをセットアップしてみました。

<!-- more -->

`init.lua`をさらしておきます。

```lua
vim.g.mapleader = ' '
vim.o.number = true
vim.o.tabstop = 4
vim.o.expandtab = true
vim.o.softtabstop = 4
vim.o.shiftwidth = 4
vim.o.swapfile = false
vim.o.cursorline = true
vim.o.scrolloff = 10
vim.o.autocomplete = true
vim.opt.completeopt = { 'menuone', 'noselect', 'popup' }
vim.o.complete = "o"
vim.o.clipboard = 'unnamedplus'

-- experimental ui
require('vim._core.ui2').enable()

-- highlight on yank
vim.api.nvim_create_autocmd('TextYankPost', {
  callback = function()
    vim.highlight.on_yank({ timeout = 300 })
  end,
})

vim.pack.add({
    'https://github.com/rebelot/kanagawa.nvim',
    'https://github.com/neovim/nvim-lspconfig',
    'https://github.com/nvim-treesitter/nvim-treesitter',
    'https://github.com/ray-x/lsp_signature.nvim', -- lsp signature hints
    'https://github.com/nvim-tree/nvim-web-devicons',
    'https://github.com/nvim-lualine/lualine.nvim',
    'https://github.com/numToStr/FTerm.nvim',
    'https://github.com/windwp/nvim-autopairs',
})

-- colorscheme setup
require('kanagawa').setup({transparent = true})
vim.cmd.colorscheme("kanagawa")

-- lsp setup
vim.lsp.config('pyrefly', {
    cmd = { 'uvx', 'pyrefly', 'lsp' },
})
vim.lsp.config('ruff', {
    cmd = { 'uvx', 'ruff', 'server' },
})
vim.lsp.config('copilot', {
    cmd = { 'npx', '@github/copilot-language-server', '--stdio' },
    settings = {
        telemetry = {
            telemetryLevel = 'off',
        },
    }
})
vim.lsp.config('lua_ls', {
    settings = {
        Lua = {
            diagnostics = {
                globals = {'vim'},
            }
        }
    },
})
local lsp_names = {
    "copilot",
    "pyrefly",
    "ruff",
    "lua_ls",
}
vim.lsp.enable(lsp_names)
vim.lsp.inline_completion.enable(true)
vim.keymap.set('i', '<Tab>', function()
  if not vim.lsp.inline_completion.get() then
    return '<Tab>'
  end
end, { expr = true, desc = 'Accept the current inline completion' })
vim.lsp.inlay_hint.enable(true)

vim.api.nvim_create_autocmd('LspAttach', {
    callback = function(args)
        local bufnr = args.buf
        local client = vim.lsp.get_client_by_id(args.data.client_id)
        if client.name == 'copilot' then
            return
        end
        local opts = { buffer = bufnr, remap = false }
        vim.keymap.set('n', 'gd', vim.lsp.buf.definition, opts)
        vim.keymap.set('n', 'K', vim.lsp.buf.hover, opts)
        vim.keymap.set('n', 'gi', vim.lsp.buf.implementation, opts)
        vim.keymap.set('n', '<C-k>', vim.lsp.buf.signature_help, opts)
        vim.keymap.set('n', '<leader>wa', vim.lsp.buf.add_workspace_folder, opts)
        vim.keymap.set('n', '<leader>wr', vim.lsp.buf.remove_workspace_folder, opts)
        vim.keymap.set('n', '<leader>wl', function()
            print(vim.inspect(vim.lsp.buf.list_workspace_folders()))
        end, opts)
        vim.keymap.set('n', '<leader>D', vim.lsp.buf.type_definition, opts)
        vim.keymap.set('n', '<leader>rn', vim.lsp.buf.rename, opts)
        vim.keymap.set({ 'n', 'v' }, '<leader>ca', vim.lsp.buf.code_action, opts)
        vim.keymap.set('n', 'gr', vim.lsp.buf.references, opts)
    end,
})

-- diagnostic setup
vim.diagnostic.config({
    underlien = true,
    virtial_text = true,
    signs = true,
    update_in_insert = false,
})

-- treesitter setup
require('nvim-treesitter').setup({})

-- lsp signature setup
require('lsp_signature').setup({
    bind = true,
    handler_opts = {
        border = "rounded"
    }
})

-- lualine setup
require('lualine').setup()

-- terminal setup
require('FTerm').setup({
    border = 'double',
    cmd = 'pwsh',
})
vim.keymap.set('n', '<A-i>', '<CMD>lua require("FTerm").toggle()<CR>')
vim.keymap.set('t', '<A-i>', '<C-\\><C-n><CMD>lua require("FTerm").toggle()<CR>')

-- autopairs setup
require('nvim-autopairs').setup({})
```
