; Inno Setup Script for Hope-AD
; AI Image Protection System

#define MyAppName "Hope-AD"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "HopeADeff"
#define MyAppURL "https://github.com/HopeADeff/Hope"
#define MyAppExeName "Hope.exe"

[Setup]
AppId={{A8F3D2E1-5B6C-4D7E-8F9A-0B1C2D3E4F5G}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=D:\After\Hope\LICENSE
OutputDir=D:\After\Hope\installer_output
OutputBaseFilename=Hope-AD-Setup-{#MyAppVersion}
SetupIconFile=D:\After\Hope\Hope\Hope\Hope.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; WPF Application
Source: "D:\After\Hope\Hope\Hope\bin\Release\net10.0-windows\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Python Engine (bundled)
Source: "D:\After\Hope\Hope\Hope\dist\engine\*"; DestDir: "{app}\engine"; Flags: ignoreversion recursesubdirs createallsubdirs
; Documentation
Source: "D:\After\Hope\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "D:\After\Hope\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "D:\After\Hope\USAGE_GUIDE.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
