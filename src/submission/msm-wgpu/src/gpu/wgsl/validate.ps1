$TempFile = Get-ChildItem ([IO.Path]::GetTempFileName()) | Rename-Item -NewName { [IO.Path]::ChangeExtension($_, ".wgsl") } -PassThru
Get-Content .\entry_padd.wgsl, .\arith.wgsl | Set-Content $TempFile
naga $TempFile

Get-Content .\entry_padd_old.wgsl, .\curve.wgsl, .\field_modulus.wgsl, .\u256.wgsl | Set-Content $TempFile
naga $TempFile

Get-Content .\entry_padd_idx.wgsl, .\curve.wgsl, .\field_modulus.wgsl, .\u256.wgsl | Set-Content $TempFile
naga $TempFile