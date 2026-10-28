import psycopg2
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:1234@localhost:5432/NeRF")


def run_database_migrations():
    conn = None
    cur = None
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        print("Starting NeRF database migrations...\n")

        cur.execute("""
            DO $$ BEGIN
                CREATE TYPE render_status AS ENUM (
                    'not_started',
                    'processing', 
                    'completed',
                    'failed'
                );
            EXCEPTION
                WHEN duplicate_object THEN NULL;
            END $$;
        """)

        print("✓ render_status enum type verified")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS render_jobs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                status render_status NOT NULL DEFAULT 'not_started',
                progress INTEGER NOT NULL DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
                gif_path VARCHAR(500),
                error_msg TEXT,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)

        print("✓ render_jobs table created/verified")

        cur.execute("""
            ALTER TABLE render_jobs 
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
        """)

        cur.execute("""
            ALTER TABLE render_jobs 
            ADD COLUMN IF NOT EXISTS input_folder_path VARCHAR(500);
        """)

        cur.execute("""
            ALTER TABLE render_jobs 
            ADD COLUMN IF NOT EXISTS model_path VARCHAR(500);
        """)

        cur.execute("""
            ALTER TABLE render_jobs 
            ADD COLUMN IF NOT EXISTS num_iterations INTEGER DEFAULT 10000;
        """)

        cur.execute("""
            ALTER TABLE render_jobs 
            ADD COLUMN IF NOT EXISTS image_size INTEGER DEFAULT 200;
        """)

        print("✓ render_jobs table columns verified")

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_render_jobs_status ON render_jobs(status);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_render_jobs_created_at ON render_jobs(created_at DESC);
        """)

        print("✓ Indexes created/verified")

        cur.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)

        cur.execute("""
            DROP TRIGGER IF EXISTS update_render_jobs_updated_at ON render_jobs;
            CREATE TRIGGER update_render_jobs_updated_at
                BEFORE UPDATE ON render_jobs
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """)

        print("✓ updated_at trigger created/verified")

        cur.execute("""
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_name = 'render_jobs'
            ORDER BY ordinal_position;
        """)
        
        columns = cur.fetchall()
        
        print("\n=== render_jobs Table Schema ===")
        print("-" * 60)
        for col in columns:
            nullable = "NULL" if col[2] == 'YES' else "NOT NULL"
            default = f" DEFAULT {col[3]}" if col[3] else ""
            print(f"  {col[0]}: {col[1]} {nullable}{default}")

        conn.commit()
        print("\n✓ All migrations completed successfully!")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def drop_tables():
    conn = None
    cur = None
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        print("Dropping NeRF tables...")
        
        cur.execute("DROP TABLE IF EXISTS render_jobs CASCADE;")
        cur.execute("DROP TYPE IF EXISTS render_status CASCADE;")
        
        conn.commit()
        print("✓ Tables dropped successfully!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "drop":
        confirm = input("Are you sure you want to drop all tables? (yes/no): ")
        if confirm.lower() == "yes":
            drop_tables()
        else:
            print("Operation cancelled.")
    else:
        run_database_migrations()
