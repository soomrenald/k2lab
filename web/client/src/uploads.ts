import { createSHA256, sha256 } from "hash-wasm";
import type { FileKind, FileRecord } from "./api";
import { controlPlane } from "./api";

export async function uploadWorkspaceFile(
  workspaceId: string,
  file: File,
  destinationKind: FileKind,
  onProgress?: (value: number) => void,
): Promise<FileRecord> {
  const hasher = await createSHA256();
  hasher.init();
  const chunkSize = 8 * 1024 * 1024;
  for (let offset = 0; offset < file.size; offset += chunkSize) {
    hasher.update(new Uint8Array(await file.slice(offset, offset + chunkSize).arrayBuffer()));
    onProgress?.(Math.min(0.1, ((offset + chunkSize) / Math.max(1, file.size)) * 0.1));
  }
  const digest = hasher.digest("hex");
  const existing = await findExisting(workspaceId, destinationKind, file.name, file.size, digest);
  if (existing) {
    onProgress?.(1);
    return existing;
  }
  const session = await controlPlane.createUpload(workspaceId, {
    filename: file.name,
    destination_kind: destinationKind,
    size_bytes: file.size,
    sha256: digest,
    chunk_size_bytes: chunkSize,
  });
  for (let index = 0; index < session.chunk_count; index += 1) {
    if (session.completed_chunks.includes(index)) continue;
    const start = index * session.chunk_size_bytes;
    const buffer = await file.slice(start, start + session.chunk_size_bytes).arrayBuffer();
    await controlPlane.uploadChunk(
      workspaceId, session.id, index, buffer, await sha256(new Uint8Array(buffer)),
    );
    onProgress?.(0.1 + ((start + buffer.byteLength) / Math.max(1, file.size)) * 0.9);
  }
  return (await controlPlane.completeUpload(workspaceId, session.id)).file;
}

async function findExisting(
  workspaceId: string,
  kind: FileKind,
  name: string,
  size: number,
  digest: string,
) {
  let cursor: string | undefined;
  do {
    const page = await controlPlane.files(workspaceId, kind, cursor);
    const match = page.items.find((item) => item.display_name === name && item.size_bytes === size && item.sha256 === digest);
    if (match) return match;
    cursor = page.next_cursor ?? undefined;
  } while (cursor);
  return undefined;
}
