import type { SVGProps } from "react";

export type IconName =
  | "spark"
  | "edit"
  | "face"
  | "cloud"
  | "folder"
  | "transfer"
  | "settings"
  | "stop"
  | "play"
  | "plus"
  | "upload"
  | "trash"
  | "chevronUp"
  | "chevronDown"
  | "check"
  | "clock"
  | "gpu"
  | "layers"
  | "sliders"
  | "wand"
  | "events";

const paths: Record<IconName, React.ReactNode> = {
  spark: <path d="m12 2 1.5 5.1L18 9l-4.5 1.9L12 16l-1.5-5.1L6 9l4.5-1.9L12 2Zm6 12 .8 2.7L21 18l-2.2 1.3L18 22l-.8-2.7L15 18l2.2-1.3L18 14ZM5 13l1 3 3 1-3 1-1 3-1-3-3-1 3-1 1-3Z" />,
  edit: <path d="M4 20h4l10.8-10.8a2.8 2.8 0 0 0-4-4L4 16v4Zm9.5-13.5 4 4M12 20h8" />,
  face: <path d="M8.5 10h.01M15.5 10h.01M9 15c1.8 1.5 4.2 1.5 6 0M4 12a8 8 0 1 0 16 0 8 8 0 1 0-16 0Zm2-4c3 0 5-1.5 6-4 1 2.5 3 4 6 4" />,
  cloud: <path d="M7 18h10a4 4 0 0 0 .7-7.9A6 6 0 0 0 6.2 9 4.5 4.5 0 0 0 7 18Z" />,
  folder: <path d="M3 6.5h7l2 2h9V19H3V6.5Z" />,
  transfer: <path d="M7 7h12m0 0-3-3m3 3-3 3M17 17H5m0 0 3 3m-3-3 3-3" />,
  settings: <path d="M12 8.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7Zm7.4 3.5 1.6 1.2-2 3.5-1.9-.8a7 7 0 0 1-2.1 1.2l-.2 2H10l-.2-2a7 7 0 0 1-2.1-1.2l-1.9.8-2-3.5L5.4 12a7 7 0 0 1 0-2.4L3.8 8.4l2-3.5 1.9.8a7 7 0 0 1 2.1-1.2l.2-2h4.8l.2 2a7 7 0 0 1 2.1 1.2l1.9-.8 2 3.5-1.6 1.2a7 7 0 0 1 0 2.4Z" />,
  stop: <path d="M7 7h10v10H7z" />,
  play: <path d="m8 5 11 7-11 7V5Z" />,
  plus: <path d="M12 5v14M5 12h14" />,
  upload: <path d="M12 16V4m0 0L7 9m5-5 5 5M5 14v6h14v-6" />,
  trash: <path d="M5 7h14M9 7V4h6v3m2 0-1 13H8L7 7m4 4v5m3-5v5" />,
  chevronUp: <path d="m7 14 5-5 5 5" />,
  chevronDown: <path d="m7 10 5 5 5-5" />,
  check: <path d="m5 12 4 4L19 6" />,
  clock: <path d="M12 7v5l3 2M4 12a8 8 0 1 0 16 0 8 8 0 1 0-16 0Z" />,
  gpu: <path d="M5 7h14v10H5V7Zm-2 3h2m-2 4h2m14-4h2m-2 4h2M9 10h6v4H9v-4Z" />,
  layers: <path d="m12 3 9 5-9 5-9-5 9-5Zm-8 9 8 4.5 8-4.5M4 16l8 4.5 8-4.5" />,
  sliders: <path d="M4 7h8m4 0h4M4 17h4m4 0h8M12 4v6M8 14v6" />,
  wand: <path d="m4 20 11-11m-3-3 6 6M6 3l.5 2L8 6l-1.5.5L6 8l-.5-1.5L4 6l1.5-1L6 3Zm13 12 .5 2 1.5.5-1.5.5-.5 2-.5-2-1.5-.5 1.5-.5.5-2Z" />,
  events: <path d="M5 5h14M5 10h14M5 15h9M5 20h7" />,
};

export function Icon({ name, ...props }: { name: IconName } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      {paths[name]}
    </svg>
  );
}
