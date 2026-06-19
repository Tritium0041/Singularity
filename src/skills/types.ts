export type SkillSource = {
  root?: string;
  kind: "directory" | "file";
};

export type Skill = {
  name: string;
  description: string;
  filePath: string;
  baseDir: string;
  disableModelInvocation: boolean;
  source: SkillSource;
};

export type SkillDiagnosticLevel = "warning" | "error";

export type SkillDiagnostic = {
  level: SkillDiagnosticLevel;
  message: string;
  path?: string;
  name?: string;
};

export type SkillLoadOptions = {
  roots?: string[];
  cwd?: string;
};

export type SkillLoadResult = {
  skills: Skill[];
  diagnostics: SkillDiagnostic[];
};
