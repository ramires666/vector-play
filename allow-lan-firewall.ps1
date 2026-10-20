# Run from elevated PowerShell on THUNDERFULL.
$ErrorActionPreference = 'Stop'
$rule = 'VectorPlay LAN TCP 58765'
$admin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { Write-Error 'Run as Administrator'; exit 1 }
Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue | Remove-NetFirewallRule
New-NetFirewallRule -DisplayName $rule -Direction Inbound -Action Allow -Enabled True -Protocol TCP -LocalPort 58765 -Profile Any -RemoteAddress @('192.168.0.0/24','192.168.66.0/24') | Out-Null
Write-Output "Installed rule: $rule (LAN only)"
