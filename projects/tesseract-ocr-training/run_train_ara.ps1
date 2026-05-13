param(
    [string]$Dataset = "E:\TRDG\new_ds_for_finetune\test_nrt_pdf_images",
    [string]$Workdir = "train\ara",
    [string]$BaseLang = "script/Arabic",
    [string]$BaseTraineddata = "train\base_models\script\Arabic.traineddata",
    [int]$MaxIterations = 2000,
    [int]$Psm = 7,
    [string]$FallbackPsm = "13",
    [switch]$ExistingLstmfOnly,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"

$argsList = @(
    "scripts\finetune.py",
    "--dataset", $Dataset,
    "--workdir", $Workdir,
    "--base-lang", $BaseLang,
    "--output-lang", "ara",
    "--base-traineddata", $BaseTraineddata,
    "--psm", "$Psm",
    "--fallback-psm", $FallbackPsm,
    "--max-iterations", "$MaxIterations"
)

if ($Limit -gt 0) {
    $argsList += @("--limit", "$Limit")
}

if ($ExistingLstmfOnly) {
    $argsList += @("--existing-lstmf-only")
}

python @argsList
